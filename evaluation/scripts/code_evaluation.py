"""Greedy HumanEval(+)/MBPP(+) generation with a separate EvalPlus CPU scorer."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluation.scripts.lora_eval_utils import (
    cleanup_eval_model_config, release_model_for_cleanup, resolve_eval_model_config,
)
from evaluation.scripts.math_evaluation import STOP_WORDS, split_rows
from evaluation.scripts.prompt_template import get_prompt_template

BRIDGE = Path(__file__).with_name("code_evalplus.py").resolve()
CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(generation, pause_token="<|PAUSE|>"):
    # Qwen stores pause as an ordinary token, so skip_special_tokens alone does
    # not remove it. Retain the raw generation separately in the output log.
    cleaned = generation.replace(pause_token, "") if pause_token else generation
    match = CODE_FENCE.search(cleaned)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"^\s*```(?:python|py)?\s*\n?", "", cleaned.strip())
    return re.sub(r"\n?\s*```\s*$", "", cleaned).strip()


def scorer_command(args, action, *options):
    return [args.evalplus_python, str(BRIDGE), action,
            "--cache-dir", str(Path(args.evalplus_cache).absolute()), *map(str, options)]


def prepare_benchmarks(args):
    command = scorer_command(args, "prepare", "--output-dir", Path(args.output_dir) / "benchmarks",
                             "--benchmarks", *args.benchmarks)
    if args.max_samples is not None:
        command += ["--max-samples", str(args.max_samples)]
    subprocess.run(command, check=True)


def sampling_kwargs(args):
    family = "qwen" if "qwen" in args.base_model.lower() else "llama"
    return dict(temperature=args.temperature, top_p=args.top_p, n=1,
                max_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                stop=STOP_WORDS[family], stop_token_ids=[151645, 151643] if family == "qwen" else None,
                skip_special_tokens=True, include_stop_str_in_output=True,
                truncate_prompt_tokens=args.prompt_max_len)


def evaluate(args):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # Preparation is CPU-only and happens before loading any model weights.
    if rank == 0:
        subprocess.run(scorer_command(args, "check"), check=True)
        prepare_benchmarks(args)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    if world > 1:
        dist.barrier()
    output = Path(args.output_dir)
    model = resolve_eval_model_config(args.pretrain, args.base_model, rank=rank, world_size=world,
                                     merged_path=str(output / ".merged_model"))
    llm = tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_path, trust_remote_code=True)
        llm = LLM(model=model.vllm_model_path, tensor_parallel_size=1, trust_remote_code=True,
                  seed=args.seed, max_num_seqs=256, enable_prefix_caching=False, enforce_eager=True,
                  max_model_len=args.max_len, distributed_executor_backend="external_launcher")
        kwargs = sampling_kwargs(args)
        sampling = SamplingParams(**kwargs)
        if rank == 0:
            (output / "evaluation_config.json").write_text(json.dumps({**vars(args), "sampling": kwargs,
                "forced_pause": False, "pause_removal": "literal text before code extraction"}, indent=2) + "\n")
        template = get_prompt_template(args.prompt_type)
        summary = {}
        for benchmark in args.benchmarks:
            tasks_path = output / "benchmarks" / f"{benchmark}.jsonl"
            rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
            local = split_rows(rows, rank, world)
            prompts = [template.format(input=row["question"]) for row in local]
            generated = llm.generate(prompts, sampling, use_tqdm=rank == 0) if prompts else []
            if len(generated) != len(local) or any(len(item.outputs) != 1 for item in generated):
                raise RuntimeError("Expected one generated solution per code task")
            with (output / f"{benchmark}_rank{rank}.jsonl").open("w") as stream:
                for row, prompt, item in zip(local, prompts, generated):
                    emitted = item.outputs[0].text
                    token_ids = list(item.outputs[0].token_ids)
                    raw = tokenizer.decode(token_ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
                    stream.write(json.dumps({"task_id": row["task_id"], "prompt": prompt,
                        "generation": raw, "generation_token_ids": token_ids,
                        "generation_text": emitted, "solution": extract_code(emitted, args.pause_token)},
                        ensure_ascii=False) + "\n")
            if world > 1:
                dist.barrier()
            result = [None]
            if rank == 0:
                try:
                    samples_path = output / f"{benchmark}.jsonl"
                    with samples_path.open("w") as stream:
                        for shard in range(world):
                            shard_path = output / f"{benchmark}_rank{shard}.jsonl"
                            stream.write(shard_path.read_text())
                            shard_path.unlink()
                    result_path = output / f"{benchmark}_evalplus.json"
                    subprocess.run(scorer_command(args, "score", "--benchmark", benchmark,
                        "--samples", samples_path, "--tasks", tasks_path, "--output", result_path,
                        "--parallel", args.evalplus_parallel), check=True)
                    summary[benchmark] = json.loads(result_path.read_text())["summary"]
                    result[0] = {"ok": True}
                except Exception as error:
                    result[0] = {"ok": False, "error": str(error)}
            if world > 1:
                dist.broadcast_object_list(result, src=0)
            if not result[0]["ok"]:
                raise RuntimeError(f"Code scoring failed: {result[0]['error']}")
        if rank == 0:
            (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            with (output / "results.csv").open("w", newline="") as stream:
                fields = ["benchmark", "questions", "pass1", "plus_pass1", "subset"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(summary.values())
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        release_model_for_cleanup(llm, tokenizer)
        llm = tokenizer = None
        cleanup_eval_model_config(model, rank=rank, world_size=world)
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_type", required=True, choices=["qwen2-code", "llama3-code"])
    parser.add_argument("--benchmarks", nargs="+", choices=["humaneval", "mbpp"], default=["humaneval", "mbpp"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--prompt_max_len", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=10240)
    parser.add_argument("--pause_token", default="<|PAUSE|>")
    parser.add_argument("--evalplus_python", default=sys.executable)
    parser.add_argument("--evalplus_parallel", type=int, default=4)
    parser.add_argument("--evalplus_cache", default="artifacts/evalplus")
    return parser


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    parsed = build_parser().parse_args()
    if parsed.max_samples is not None and parsed.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if parsed.evalplus_parallel < 1:
        raise ValueError("evalplus_parallel must be positive")
    evaluate(parsed)
