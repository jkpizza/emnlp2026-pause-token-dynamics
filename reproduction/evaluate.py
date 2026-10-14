"""Launch math, code, or four-shot general benchmark evaluation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
import subprocess

from reproduction.common import ROOT, cli_args, gpu_count, launch, load_yaml, torchrun

MODELS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4b": "Qwen/Qwen3-4B-Base",
    "qwen3-8b": "Qwen/Qwen3-8B-Base",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
CODE_RECIPES = {"qwen3-1.7b", "qwen3-4b", "qwen3-8b", "llama3.1-8b"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, choices=MODELS)
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft")
    parser.add_argument("--checkpoint", required=True, help="HF model ID, full checkpoint, or local LoRA adapter")
    parser.add_argument("--suite", choices=["math", "code", "general"], default="math")
    parser.add_argument("--method", default=None, help="Optional method label recorded in the invocation")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Evaluate two examples per benchmark or general task")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", help="Math benchmark stems or code benchmarks humaneval/mbpp")
    parser.add_argument("--tasks", help="Optional LightEval task string")
    parser.add_argument("--merge-python", help="Training-environment Python for general evaluation of a local LoRA adapter")
    parser.add_argument("--evalplus-python", help="Python containing EvalPlus 0.3.1 for code evaluation; defaults to this interpreter")
    return parser


def command_for(args):
    if args.suite == "code" and (args.stage != "sft" or args.recipe not in CODE_RECIPES):
        raise ValueError("Code evaluation supports SFT with qwen3-1.7b, qwen3-4b, qwen3-8b, and llama3.1-8b")
    if args.stage == "grpo" and args.recipe not in ("qwen3-1.7b", "qwen3-4b"):
        raise ValueError("GRPO recipes are available for qwen3-1.7b and qwen3-4b")
    if args.tasks and args.suite != "general":
        raise ValueError("--tasks is available for the general suite")
    if args.benchmarks and args.suite not in ("math", "code"):
        raise ValueError("--benchmarks is available for the math and code suites")
    if args.evalplus_python and args.suite != "code":
        raise ValueError("--evalplus-python is available for the code suite")
    if args.suite == "code" and args.benchmarks and not set(args.benchmarks) <= {"humaneval", "mbpp"}:
        raise ValueError("Code benchmarks must be humaneval or mbpp")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.checkpoint).strip("_") or "checkpoint"
    if len(label) > 100:
        label = label[:80] + "_" + hashlib.sha256(args.checkpoint.encode()).hexdigest()[:12]
    output = args.output_dir or Path("outputs/evaluation") / args.stage / args.recipe / label / args.suite
    # Command inspection must not invoke GPU discovery or other subprocesses.
    num_gpus = args.num_gpus if args.num_gpus is not None else (1 if args.dry_run else gpu_count())
    if num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    checkpoint = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint
    config = load_yaml(f"configs/evaluation/{args.suite}.yaml")
    if args.suite == "math":
        stages = config.pop("stages")
        config.update(stages[args.stage])
        config.update(stage=args.stage, pretrain=checkpoint, base_model=MODELS[args.recipe],
                      prompt_type="llama3" if args.recipe.startswith("llama") else "qwen2-boxed-cot",
                      output_dir=str(output), max_samples=2 if args.smoke else None,
                      benchmarks=args.benchmarks)
        command = torchrun("evaluation.scripts.math_evaluation", cli_args(config), num_gpus)
    elif args.suite == "code":
        scorer_python = args.evalplus_python or sys.executable
        if Path(scorer_python).exists():
            scorer_python = str(Path(scorer_python).absolute())
        config.update(pretrain=checkpoint, base_model=MODELS[args.recipe],
                      prompt_type="llama3-code" if args.recipe.startswith("llama") else "qwen2-code",
                      output_dir=str(output), max_samples=2 if args.smoke else None,
                      benchmarks=args.benchmarks or ["humaneval", "mbpp"], evalplus_python=scorer_python)
        if args.smoke:
            config["max_new_tokens"] = 256
        command = torchrun("evaluation.scripts.code_evaluation", cli_args(config), num_gpus)
    else:
        config.update(pretrain=checkpoint, base_model=MODELS[args.recipe], stage=args.stage,
                      output_dir=str(output), num_gpus=num_gpus,
                      max_samples=2 if args.smoke else None, merge_python=args.merge_python)
        if args.tasks:
            config["tasks"] = args.tasks
        command = [sys.executable, "-m", "evaluation.scripts.general_evaluation", *cli_args(config)]
    return command, output, config


def main(argv=None):
    args = build_parser().parse_args(argv)
    command, output, config = command_for(args)
    if args.suite == "code" and not args.dry_run:
        bridge = ROOT / "evaluation/scripts/code_evalplus.py"
        # Check dependencies before torchrun, CUDA initialization, or loading weights.
        subprocess.run([config["evalplus_python"], str(bridge), "check"], check=True, cwd=ROOT)
    launch(command, output, dry_run=args.dry_run,
           metadata={"recipe": args.recipe, "stage": args.stage, "method": args.method,
                     "suite": args.suite, "smoke": args.smoke, "parameters": config})


if __name__ == "__main__":
    main()
