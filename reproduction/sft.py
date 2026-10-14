"""Separate dataset construction and SFT commands backed by explicit recipes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reproduction.common import ROOT, cli_args, gpu_count, launch, load_yaml, torchrun

RECIPES = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b", "llama3.2-1b", "llama3.2-3b", "llama3.1-8b")
CODE_RECIPES = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b", "llama3.1-8b")
METHODS = ("none", "step", "append", "random", "dit")


def recipe_config(recipe, domain="math"):
    if domain == "code" and recipe not in CODE_RECIPES:
        raise ValueError(f"Code SFT supports: {', '.join(CODE_RECIPES)}.")
    prefix = "code/" if domain == "code" else ""
    return load_yaml(f"configs/sft/{prefix}{recipe}.yaml")


def dataset_directory(args):
    if args.dataset_dir:
        return Path(args.dataset_dir).resolve()
    suffix = "_smoke" if args.smoke else ""
    root = ROOT / "artifacts" / "datasets"
    if getattr(args, "domain", "math") == "code":
        root /= "code"
    return root / args.recipe / (args.method + suffix)


def dataset_settings(args):
    domain = getattr(args, "domain", "math")
    config = recipe_config(args.recipe, domain)
    directory = dataset_directory(args)
    default_source = args.dataset is None
    source = config["dataset"]["repo_id"] if default_source else args.dataset
    if Path(source).exists():
        source = str(Path(source).resolve())
    limit = args.max_samples
    if limit is not None and limit < 1:
        raise ValueError("--max-samples must be positive.")
    if args.smoke:
        limit = min(limit, 32) if limit is not None else 32
    settings = {
        "dataset": source, "dataset_split": args.split,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision or (config["dataset"]["revision"] if default_source else None),
        "max_samples": limit, "tokenizer_path": config["model"],
        "pause_tokenizer_save_path": str(directory / "tokenizer"), "save_path": str(directory),
        "method": "code_line" if domain == "code" and args.method == "step" else args.method,
        "budget_mode": "exact_steps", "pause_tokens_per_step": 1,
        "dit_model_path": config["model"],
        "input_key": args.input_key or (config["dataset"].get("input_key", "query") if default_source else "question"),
        "output_key": args.output_key or (config["dataset"].get("output_key", "response") if default_source else "answer"),
        "seed": 42, "eval_split_ratio": 0.05, "max_length": config["training"]["max_len"],
        "split_mode": "sentence", "splitter": "code" if domain == "code" else "atomic_math",
        "expected_pause_token_id": config["dataset"]["pause_token_id"],
        "pause_tokenizer_single_word": config["dataset"]["tokenizer_single_word"],
        "pause_tokenizer_special": config["dataset"]["tokenizer_special"],
    }
    if domain == "code":
        settings["domain"] = domain
    return settings, directory


def build_dataset(args):
    settings, directory = dataset_settings(args)
    command = [sys.executable, "-m", "openrlhf.cli.build_pause_dataset", *cli_args(settings)]
    launch(command, output_dir=directory, dry_run=args.dry_run,
           metadata={"domain": getattr(args, "domain", "math"), "recipe": args.recipe,
                     "method": args.method, "smoke": args.smoke, "settings": settings})


def training_settings(args):
    domain = getattr(args, "domain", "math")
    config = recipe_config(args.recipe, domain)
    if args.predict_pause_tokens and args.method != "step":
        raise ValueError("--predict-pause-tokens is the step ablation; use --method step.")
    variant = "step_unmasked" if args.predict_pause_tokens else args.method
    variant_config = config["variants"][variant]
    settings = {**config["training"], **variant_config.get("training", {})}
    settings["max_ckpt_mem"] = int(float(settings.get("max_ckpt_mem", 1e8)))
    directory = dataset_directory(args)
    output_root = ROOT / "outputs" / "sft"
    if domain == "code":
        output_root /= "code"
    output = Path(args.output_dir).resolve() if args.output_dir else output_root / args.recipe / (variant + ("_smoke" if args.smoke else ""))
    settings.update({"dataset": str(directory / "train_pause_dataset.jsonl"),
                     "input_key": "question", "output_key": "answer",
                     "dataset_split": "train", "eval_split": "train",
                     "save_path": str(output), "ckpt_path": str(output / "ckpt"),
                     "save_hf_ckpt": True, "disable_ds_ckpt": True,
                     "pause_token": "<|PAUSE|>",
                     "wandb_run_name": f"{'code-' if domain == 'code' else ''}{args.recipe}-{variant}"})
    if variant_config.get("holdout_eval"):
        settings["eval_dataset"] = str(directory / "eval_pause_dataset.jsonl")
    if domain == "code":
        settings["pause_tokenizer_save_path"] = str(directory / "tokenizer")
    if args.wandb:
        settings.update(use_wandb="true", wandb_project="pause-token-sft")
    count = gpu_count(args.num_gpus if args.num_gpus is not None else (1 if args.dry_run else None))
    if args.smoke:
        settings.update(max_steps=1, max_epochs=1, max_samples=max(32, count * 2),
                        train_batch_size=count, micro_train_batch_size=1, max_len=256,
                        save_steps=1, logging_steps=1, eval_steps=-1)
        if domain == "code":
            settings["lr_warmup_ratio"] = 0.0
        settings.pop("eval_dataset", None)
    batch = int(settings["train_batch_size"])
    micro = int(settings["micro_train_batch_size"])
    if batch < 1 or micro < 1 or batch % (micro * count):
        raise ValueError(f"Global batch {batch} must be divisible by micro batch {micro} × GPUs {count}.")
    metadata = {"domain": domain, "recipe": args.recipe, "method": args.method, "variant": variant,
                "settings": settings, "num_gpus": count,
                "smoke": args.smoke}
    return settings, output, metadata


def train(args):
    settings, output, metadata = training_settings(args)
    if not args.dry_run:
        metadata_path = dataset_directory(args) / "metadata.json"
        if metadata_path.is_file():
            dataset_metadata = json.loads(metadata_path.read_text())
            dataset_domain = dataset_metadata.get("domain", "math")
            if dataset_domain != metadata["domain"]:
                raise ValueError(f"Dataset domain {dataset_domain!r} does not match --domain {metadata['domain']}; "
                                 "use the matching dataset directory or rebuild it.")
            if dataset_domain == "code":
                expected_id = recipe_config(args.recipe, "code")["dataset"]["pause_token_id"]
                if dataset_metadata.get("pause_token_id", expected_id) != expected_id:
                    raise ValueError("Dataset tokenizer does not match the selected recipe; rebuild the data "
                                     "with the same --recipe used for training.")
                if dataset_metadata.get("method", args.method) != args.method:
                    raise ValueError("Dataset placement does not match --method; use the same method for "
                                     "building and training.")
        for key in ("dataset", "eval_dataset"):
            if key in settings and not Path(settings[key]).is_file():
                raise FileNotFoundError(f"Missing {settings[key]}; run scripts/build_dataset.sh first.")
        # Very small custom datasets can have no 5% holdout rows.
        if settings.get("eval_dataset") and Path(settings["eval_dataset"]).stat().st_size == 0:
            settings.pop("eval_dataset")
    command = torchrun("openrlhf.cli.train_pause_sft", cli_args(settings), metadata["num_gpus"])
    launch(command, output_dir=output, dry_run=args.dry_run, metadata=metadata)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build-dataset", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--recipe", "--model", choices=RECIPES, required=True)
        command.add_argument("--domain", choices=("math", "code"), default="math")
        command.add_argument("--method", choices=METHODS, default="step")
        command.add_argument("--dataset-dir", type=Path)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--smoke", action="store_true")
        if name == "train":
            command.add_argument("--predict-pause-tokens", action="store_true")
            command.add_argument("--output-dir", type=Path)
            command.add_argument("--num-gpus", type=int)
            command.add_argument("--wandb", action="store_true")
        else:
            command.add_argument("--dataset", help="HF dataset ID, local JSONL/JSON/CSV/Parquet, or saved dataset directory")
            command.add_argument("--split", default="train")
            command.add_argument("--dataset-config")
            command.add_argument("--dataset-revision")
            command.add_argument("--max-samples", type=int)
            command.add_argument("--input-key", help="Prompt column; bundled math/code: query/problem; custom: question")
            command.add_argument("--output-key", help="Response column; bundled math/code: response/solution; custom: answer")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        (build_dataset if args.command == "build-dataset" else train)(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
