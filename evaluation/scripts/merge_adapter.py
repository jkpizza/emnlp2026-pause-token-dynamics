"""Prepare an evaluation-only LoRA merge using the training environment."""
import argparse
from evaluation.scripts.lora_eval_utils import (
    adapter_base_model_path, has_tokenizer_files, merge_lora_adapter_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = args.base_model or adapter_base_model_path(args.adapter)
    if not base:
        parser.error("Adapter has no base_model_name_or_path; provide --base-model")
    merge_lora_adapter_checkpoint(args.adapter, base, args.output,
                                  args.adapter if has_tokenizer_files(args.adapter) else base)


if __name__ == "__main__":
    main()
