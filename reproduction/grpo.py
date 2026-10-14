"""Train normal or post-rollout pause GRPO on mathematical question/answer pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

from reproduction.common import ROOT, cli_args, gpu_count, launch, load_yaml

RECIPES = ("qwen3-1.7b", "qwen3-4b")
METHODS = ("normal", "pause")


def build_settings(
    recipe,
    method,
    num_gpus=None,
    output_dir=None,
    smoke=False,
    *,
    dataset=None,
    split=None,
    dataset_config=None,
    dataset_revision=None,
    max_samples=None,
    input_key=None,
    label_key=None,
    no_eval=False,
):
    config = load_yaml(f"configs/grpo/{recipe}-{method}.yaml")
    settings = dict(config["settings"])
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max-samples must be a positive integer.")
    settings.update(
        prompt_data=str(dataset) if dataset is not None else settings["prompt_data"],
        prompt_split=split or settings.get("prompt_split", "train"),
        prompt_data_config=dataset_config,
        prompt_data_revision=dataset_revision,
        input_key=input_key
        or ("question" if dataset is not None else settings["input_key"]),
        label_key=label_key
        or ("answer" if dataset is not None else settings["label_key"]),
        max_samples=max_samples,
    )
    # Use the recorded topology unless the user explicitly changes the GPU count.
    count = (
        gpu_count(num_gpus)
        if num_gpus is not None
        else settings["actor_num_gpus_per_node"]
    )
    output = (
        Path(output_dir).resolve()
        if output_dir
        else (
            ROOT / "outputs" / "grpo" / recipe / (method + ("_smoke" if smoke else ""))
        )
    )
    for role in ("actor", "ref"):
        settings[f"{role}_num_nodes"] = 1
        settings[f"{role}_num_gpus_per_node"] = count
    settings.update(
        vllm_num_engines=count,
        vllm_tensor_parallel_size=1,
        save_path=str(output / "final"),
        ckpt_path=str(output / "ckpt"),
        pause_tokenizer_save_path=str(output / "tokenizer")
        if method == "pause"
        else None,
        use_wandb="true",
        wandb_project="pause-token-reproduction",
        wandb_run_name=f"{recipe}-{method}" + ("-smoke" if smoke else ""),
    )
    if smoke:
        # One rollout batch, one epoch and one optimizer update; no benchmark download.
        settings.update(
            max_updates=1,
            max_samples=min(max_samples, 2 * count)
            if max_samples is not None
            else 2 * count,
            rollout_batch_size=2 * count,
            vllm_generate_batch_size=2 * count,
            n_samples_per_prompt=2,
            micro_train_batch_size=1,
            micro_rollout_batch_size=1,
            train_batch_size=4 * count,
            max_epochs=1,
            num_episodes=1,
            prompt_max_len=256,
            generate_max_len=128,
            max_len=None,
            eval_dataset=None,
            eval_dataset2=None,
            eval_steps=-1,
            save_steps=-1,
        )
    if no_eval:
        settings.update(eval_dataset=None, eval_dataset2=None, eval_steps=-1)
    return config, settings, output


def prepare_data(args):
    """Validate all input data before Ray starts and snapshot the selected rows."""
    from reproduction.data import load_source, validate_columns

    train = load_source(
        args.prompt_data,
        split=args.prompt_split,
        config=args.prompt_data_config,
        revision=args.prompt_data_revision,
        max_samples=args.max_samples,
    )
    if (
        args.prompt_data == "saramal/SimpleRL-Zoo-Data-random-5K"
        and args.label_key == "label"
        and "label" in train.column_names
    ):
        # The public default contains one question with an unextracted answer.
        # Keep custom data validation strict instead of silently filtering it.
        keep = [
            index
            for index, label in enumerate(train["label"])
            if not isinstance(label, str) or label.strip()
        ]
        skipped = len(train) - len(keep)
        if skipped:
            source_count = len(train)
            train = train.select(keep)
            warnings.warn(
                f"Skipping {skipped} missing-answer row(s) from the default SimpleRL dataset: "
                f"{source_count:,} source rows, {len(train):,} usable training rows.",
                stacklevel=2,
            )
    validate_columns(train, args.input_key, args.label_key)
    batch_size = args.vllm_generate_batch_size or args.rollout_batch_size
    if len(train) < batch_size:
        raise ValueError(
            f"GRPO needs at least {batch_size} training rows for one rollout batch; "
            f"the selected dataset has {len(train)}. Increase --max-samples or supply "
            "more data. Use --smoke for a small validation run."
        )
    sources = [("prompt_data", "train", train, args.input_key, args.label_key)]
    for suffix in ("", "2"):
        source = getattr(args, f"eval_dataset{suffix}")
        if source:
            data = load_source(source, split=getattr(args, f"eval_split{suffix}"))
            question_key = getattr(args, f"eval_input_key{suffix}")
            answer_key = getattr(args, f"eval_label_key{suffix}")
            validate_columns(data, question_key, answer_key)
            sources.append(
                (
                    f"eval_dataset{suffix}",
                    f"eval{suffix}",
                    data,
                    question_key,
                    answer_key,
                )
            )

    directory = Path(args.save_path).resolve().parent / "data"
    directory.mkdir(parents=True, exist_ok=True)
    for field, name, data, question_key, answer_key in sources:
        target = directory / f"{name}.jsonl"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                for row in data:
                    record = {
                        question_key: row[question_key],
                        answer_key: row[answer_key],
                    }
                    if "datasource" in row:
                        record["datasource"] = row["datasource"]
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        setattr(args, field, str(target))
    args.prompt_split = "train"
    args.prompt_data_config = None
    args.prompt_data_revision = None
    args.eval_split = args.eval_split2 = "train"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--dataset",
        help="Local JSONL path or Hugging Face dataset ID; defaults to the recipe dataset",
    )
    parser.add_argument("--split", help="Hugging Face split (default: train)")
    parser.add_argument("--dataset-config", help="Hugging Face dataset configuration")
    parser.add_argument("--dataset-revision", help="Hugging Face dataset revision")
    parser.add_argument(
        "--max-samples", type=int, help="Limit training rows; by default use all rows"
    )
    parser.add_argument(
        "--input-key", help="Question field (custom data: question; recipe data: text)"
    )
    parser.add_argument(
        "--label-key",
        help="Final-answer field (custom data: answer; recipe data: label)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable training-time GSM8K and MATH500 evaluation",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Override recorded 4-GPU (1.7B) or 2-GPU (4B) topology",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable online W&B using existing/environment credentials",
    )
    args = parser.parse_args(argv)
    _, settings, output = build_settings(
        args.recipe,
        args.method,
        args.num_gpus,
        args.output_dir,
        args.smoke,
        dataset=args.dataset,
        split=args.split,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        max_samples=args.max_samples,
        input_key=args.input_key,
        label_key=args.label_key,
        no_eval=args.no_eval,
    )
    os.environ["WANDB_MODE"] = "online" if args.wandb else "offline"
    os.environ.setdefault("RAY_INCLUDE_DASHBOARD", "0")
    os.environ.setdefault("RAY_USE_RANDOM_PORTS", "1")
    command = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_pause_ppo_ray",
        *cli_args(settings),
    ]
    launch(
        command,
        output_dir=output,
        dry_run=args.dry_run,
        metadata={
            "recipe": args.recipe,
            "method": args.method,
            "smoke": args.smoke,
            "wandb_mode": os.environ["WANDB_MODE"],
            "settings": settings,
        },
    )


if __name__ == "__main__":
    main()
