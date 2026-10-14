"""Recompute the selected GRPO AUCs from bundled curves or W&B."""
import argparse
from pathlib import Path
import sys
from reproduction.common import launch, load_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auc"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.source == "csv":
        command = [sys.executable, "-m", "wandb_auc_analysis.analyze_csv_auc", "--out-dir", str(args.output_dir)]
    else:
        config = load_yaml("configs/evaluation/auc.yaml")
        command = [sys.executable, "-m", "wandb_auc_analysis.analyze_wandb_auc",
                   "--runs", *config["runs"].values(), "--metrics", *config["metrics"],
                   "--step-key", "global_step", "--out-dir", str(args.output_dir),
                   "--max-step", "78"]
    launch(command, args.output_dir, dry_run=args.dry_run, metadata={"source": args.source})


if __name__ == "__main__":
    main()
