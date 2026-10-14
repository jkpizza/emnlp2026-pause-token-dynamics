"""Run the original four-shot LightEval tasks and save compact result tables."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.scripts.lora_eval_utils import (
    cleanup_temp_merged_model_path, is_lora_adapter_checkpoint,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pretrain", "base_model", "output_dir", "tasks"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft")
    parser.add_argument("--merge_python")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_model_length", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = args.pretrain
    merged = None
    try:
        if is_lora_adapter_checkpoint(model):
            if not args.merge_python and importlib.util.find_spec("peft") is None:
                raise SystemExit("This is a LoRA adapter. Add --merge-python /path/to/training/env/bin/python "
                                 "to scripts/evaluate.sh, or pass an already merged checkpoint.")
            merged = output / ".merged_model"
            subprocess.run([args.merge_python or sys.executable, "-m", "evaluation.scripts.merge_adapter",
                            "--adapter", model, "--base-model", args.base_model,
                            "--output", str(merged)], check=True)
            model = str(merged)
        environment = os.environ.copy()
        environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        environment.setdefault("GIT_PYTHON_REFRESH", "quiet")
        # Historical general evaluation never inserted or banned pause tokens,
        # including when the supplied checkpoint was trained with GRPO.
        environment["PAUSE_EVAL_BAN_TOKEN"] = "0"
        settings = {
            "model_name": model, "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_length": args.max_model_length,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs, "trust_remote_code": "true",
            "data_parallel_size": args.num_gpus, "seed": args.seed,
            "generation_parameters": "{temperature:" + str(args.temperature) + "}",
        }
        model_args = ",".join(f"{key}={value}" for key, value in settings.items())
        command = [sys.executable, "-m", "evaluation.scripts.run_lighteval_vllm", "vllm", model_args,
                   args.tasks, "--output-dir", str(output / "lighteval"),
                   "--dataset-loading-processes", "1", "--num-fewshot-seeds", "1", "--job-id", "0",
                   "--custom-tasks", "evaluation/scripts/lighteval_custom_tasks.py"]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        started = time.time()
        subprocess.run(command, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "evaluation.scripts.export_lighteval_results",
                        "--output-root", str(output / "lighteval"), "--model", model, "--tasks", args.tasks,
                        "--display-model", args.pretrain,
                        "--csv", str(output / "results.csv"), "--txt", str(output / "results.txt"),
                        "--min-mtime", str(started)], check=True)
    finally:
        if merged is not None and merged.exists():
            cleanup_temp_merged_model_path(str(merged))


if __name__ == "__main__":
    main()
