"""Math benchmark evaluation with the historical SFT and GRPO decoding rules."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from evaluation.scripts.lora_eval_utils import (
    cleanup_eval_model_config, release_model_for_cleanup, resolve_eval_model_config,
)
from evaluation.scripts.prompt_template import get_prompt_template

STOP_WORDS = {
    "qwen": ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:", "<|end|>"],
    "llama": ["</s>", "<|endoftext|>", "\n\nQuestion:"],
}


def split_rows(rows, rank, world_size):
    size, remainder = divmod(len(rows), world_size)
    start = rank * size + min(rank, remainder)
    return rows[start : start + size + (rank < remainder)]


def sampling_kwargs(args, pause_token_id=None):
    """Keep SFT text prompts and GRPO tokenized prompts/stop rules distinct."""
    kwargs = dict(temperature=args.temperature, top_p=args.top_p,
                  max_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                  include_stop_str_in_output=True, logprobs=None)
    if args.stage == "sft":
        family = "qwen" if "qwen" in args.base_model.lower() else "llama"
        kwargs.update(stop=STOP_WORDS[family], skip_special_tokens=True, n=args.eval_k,
                      stop_token_ids=[151645, 151643] if family == "qwen" else None,
                      truncate_prompt_tokens=args.prompt_max_len)
    else:
        kwargs.update(skip_special_tokens=False, top_k=-1, min_tokens=0)
        if pause_token_id is not None:
            # The pinned vLLM supports both. Never silently drop the ban.
            kwargs.update(bad_words=[args.pause_token], logit_bias={pause_token_id: -100.0})
    return kwargs


def single_pause_id(tokenizer, token):
    if token in tokenizer.get_vocab():
        return int(tokenizer.get_vocab()[token])
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False) == token:
        return int(ids[0])
    return None


def generate_groups(llm, tokenizer, prompts, sampling, args, rank=0):
    if not prompts:
        return []
    if args.stage == "sft":
        outputs = llm.generate(prompts, sampling, use_tqdm=rank == 0)
        groups = [[sample.text for sample in output.outputs] for output in outputs]
    else:
        from vllm.inputs import TokensPrompt

        expanded = [prompt for prompt in prompts for _ in range(args.eval_k)]
        ids = tokenizer(expanded, add_special_tokens=False, max_length=args.prompt_max_len,
                        truncation=True)["input_ids"]
        outputs = llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(row)) for row in ids],
                               sampling_params=sampling, use_tqdm=rank == 0)
        if len(outputs) != len(expanded):
            raise RuntimeError("GRPO generation count does not match the requested samples")
        texts = [out.outputs[0].text if out.outputs else "" for out in outputs]
        groups = [texts[i:i + args.eval_k] for i in range(0, len(texts), args.eval_k)]
    if len(groups) != len(prompts) or any(len(group) != args.eval_k for group in groups):
        raise RuntimeError("Generation count does not match the evaluation questions/samples")
    return groups


def evaluate(args):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from evaluation.scripts.scoring import score_generation_group

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    model = resolve_eval_model_config(args.pretrain, args.base_model, rank=rank, world_size=world,
                                     merged_path=str(Path(args.output_dir) / ".merged_model"))
    llm = tokenizer = None
    output = Path(args.output_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_path, trust_remote_code=True)
        if args.stage == "grpo":
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        model_args = dict(model=model.vllm_model_path, tensor_parallel_size=1, trust_remote_code=True,
                          seed=args.seed, max_num_seqs=256, enable_prefix_caching=False,
                          enforce_eager=True, distributed_executor_backend="external_launcher")
        if args.stage == "grpo":
            model_args["max_model_len"] = args.max_len
        llm = LLM(**model_args)
        pause_id = single_pause_id(tokenizer, args.pause_token)
        kwargs = sampling_kwargs(args, pause_id)
        sampling = SamplingParams(**kwargs)
        output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (output / "evaluation_config.json").write_text(
                json.dumps({**vars(args), "sampling": kwargs, "pause_token_id": pause_id}, indent=2) + "\n")
        files = sorted(Path(args.dataset).glob("*.jsonl"))
        if args.benchmarks:
            requested = set(args.benchmarks)
            available = {path.stem for path in files}
            if not requested <= available:
                raise ValueError(f"Unknown benchmarks: {sorted(requested - available)}")
            files = [path for path in files if path.stem in requested]
        if not files:
            raise ValueError(f"No math benchmark files in {args.dataset}")
        summary = {}
        template = get_prompt_template(args.prompt_type)
        for path in files:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if args.max_samples is not None:
                rows = rows[:args.max_samples]
            local = split_rows(rows, rank, world)
            prompts = [template.format(input=row["question"]) for row in local]
            groups = generate_groups(llm, tokenizer, prompts, sampling, args, rank)
            correct = 0
            rank_file = output / f"{path.stem}_rank{rank}.jsonl"
            with rank_file.open("w", encoding="utf-8") as stream:
                for row, prompt, generations in zip(local, prompts, groups):
                    result = score_generation_group(generations, row["answer"], "The answer is:", "pass")
                    correct += int(result["is_correct"])
                    stream.write(json.dumps({"data_id": row["index"], "prompt": prompt,
                        "generations": generations, "gold_answer": row["answer"], **result}, ensure_ascii=False) + "\n")
            counts = [None] * world
            if world > 1:
                dist.all_gather_object(counts, {"questions": len(local), "correct": correct})
            else:
                counts = [{"questions": len(local), "correct": correct}]
            if rank == 0:
                total = sum(item["questions"] for item in counts)
                total_correct = sum(item["correct"] for item in counts)
                summary[path.stem] = {"questions": total, "correct": total_correct,
                                      "accuracy": total_correct / total if total else 0.0}
                with (output / f"{path.stem}.jsonl").open("w", encoding="utf-8") as merged:
                    for shard in range(world):
                        shard_path = output / f"{path.stem}_rank{shard}.jsonl"
                        merged.write(shard_path.read_text())
                        shard_path.unlink()
        if rank == 0:
            (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            with (output / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["benchmark", "questions", "correct", "accuracy"])
                writer.writeheader()
                writer.writerows({"benchmark": name, **metrics} for name, metrics in summary.items())
            print(json.dumps(summary, indent=2))
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
    parser.add_argument("--stage", required=True, choices=["sft", "grpo"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="evaluation/eval_data")
    parser.add_argument("--prompt_type", choices=["qwen2-boxed-cot", "llama3"], required=True)
    parser.add_argument("--pause_token", default="<|PAUSE|>")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--prompt_max_len", type=int, required=True)
    parser.add_argument("--max_len", type=int, default=3072)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--eval_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", nargs="+")
    return parser


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    parsed = build_parser().parse_args()
    if parsed.eval_k < 1 or (parsed.eval_k > 1 and parsed.temperature == 0):
        raise ValueError("eval_k must be positive; multiple samples require temperature > 0")
    evaluate(parsed)
