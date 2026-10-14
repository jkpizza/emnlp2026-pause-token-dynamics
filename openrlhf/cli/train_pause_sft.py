import argparse
import math
import os
from datetime import datetime

from peft import PeftModelForCausalLM
from transformers.trainer import get_scheduler

from openrlhf.datasets import PauseSFTDataset
from reproduction.data import load_source, validate_columns
from openrlhf.models import PauseLM
from openrlhf.models.pause_lm import ensure_pause_tokenizer
from openrlhf.trainer.pause_sft_trainer import PauseSFTTrainer
from openrlhf.utils import get_strategy


def validate_batch(args, sample_count, world_size):
    batch = args.train_batch_size
    micro = args.micro_train_batch_size
    parallel = args.ring_attn_size * args.ds_tensor_parallel_size
    if min(batch, micro, world_size, parallel) < 1 or world_size % parallel:
        raise ValueError("Batch sizes and GPU counts must be positive, and parallel dimensions must divide the GPU count.")
    workers = world_size // parallel
    if batch % (micro * workers):
        raise ValueError(f"Global batch {batch} must be divisible by micro batch {micro} × data-parallel GPUs {workers}.")
    if sample_count < batch:
        raise ValueError(f"Only {sample_count} training samples are available; global batch {batch} requires at least "
                         f"{batch}. Use more examples, reduce the recipe batch size, or use --smoke for a quick check.")


def load_training_data(args):
    train_data = load_source(args.dataset, split=args.dataset_split, config=args.dataset_config,
                             revision=args.dataset_revision, max_samples=args.max_samples)
    validate_columns(train_data, args.input_key, args.output_key)
    validate_batch(args, len(train_data), int(os.environ.get("WORLD_SIZE", "1")))
    eval_data = None
    if args.eval_dataset:
        eval_data = load_source(args.eval_dataset, split=args.eval_split)
        validate_columns(eval_data, args.input_key, args.output_key)
    return train_data, eval_data


def train(args):
    # Invalid columns, empty datasets, and incompatible batches fail before
    # accelerator setup or loading any model weights.
    train_data, eval_data = load_training_data(args)
    strategy = get_strategy(args)
    strategy.setup_distributed()
    tokenizer = ensure_pause_tokenizer(
        args.pretrain, args.pause_tokenizer_save_path or args.save_path,
        pause_token=args.pause_token, use_fast=not args.disable_fast_tokenizer,
        padding_side="right", print_fn=strategy.print,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': [args.pause_token]})
    train_dataset = PauseSFTDataset(
        train_data, tokenizer, args.max_len, strategy,
        pretrain_mode=args.pretrain_mode, input_template=args.input_template,
        prompt_type=args.prompt_type, multiturn=args.multiturn,
    )
    # Prompt formatting can discard overlong examples. Check the resulting
    # dataset while only the tokenizer, rather than the full model, is loaded.
    validate_batch(args, len(train_dataset), int(os.environ.get("WORLD_SIZE", "1")))
    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn,
    )
    eval_dataloader = None
    if eval_data is not None:
        eval_dataset = PauseSFTDataset(
            eval_data, tokenizer, args.max_len, strategy,
            pretrain_mode=args.pretrain_mode, input_template=args.input_template,
            prompt_type=args.prompt_type, multiturn=args.multiturn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn,
        )

    model = PauseLM(
        args.pretrain,
        strategy=strategy,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        padding_side="right",
        pause_token=args.pause_token,
        pause_init_method=args.pause_init_method,
        pause_tokenizer_save_path=args.pause_tokenizer_save_path or args.save_path,
        use_fast=not args.disable_fast_tokenizer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        lora_embedding_train_mode=args.lora_embedding_train_mode,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    tokenizer = model.get_tokenizer()
    strategy.print(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    if args.max_steps > 0:
        max_steps = min(max_steps, args.max_steps)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    model, optim, scheduler = strategy.prepare((model, optim, scheduler))

    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    input_ids, _, _ = next(iter(train_dataloader))
    strategy.print("Dataset example:")
    strategy.print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

    trainer = PauseSFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    strategy.save_model(model, tokenizer, args.save_path)
    tokenizer.save_pretrained(args.save_path)

    if args.save_merged and isinstance(strategy._unwrap_model(model), PeftModelForCausalLM):
        strategy.print("\nSave merged model...\n")
        from torch import distributed as dist
        import torch

        merged_save_path = args.save_path + "_merged"
        if strategy.is_rank_0():
            os.makedirs(merged_save_path, exist_ok=True)
            tokenizer.save_pretrained(merged_save_path)
            model_to_merge = strategy._unwrap_model(model)
            merged_model = model_to_merge.merge_and_unload()
            merged_model.save_pretrained(save_directory=merged_save_path)
            output_config_file = os.path.join(merged_save_path, "config.json")
            merged_model.config.to_json_file(output_config_file)
            with open("last_model_path.txt", "w", encoding="utf-8") as f:
                f.write(merged_save_path)
        dist.barrier()
        torch.cuda.synchronize()


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pause_token", type=str, default="<|PAUSE|>")
    parser.add_argument("--pause_init_method", type=str, default="mean", choices=["mean", "gaussian_norm", "none"])
    parser.add_argument("--pause_tokenizer_save_path", type=str, default=None)
    parser.add_argument("--predict_pause_tokens", action="store_true", default=False)

    parser.add_argument("--save_path", type=str, default="./checkpoints/pause_sft")
    parser.add_argument("--save_merged", action="store_true", default=False)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/pause_sft/ckpt")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer updates; -1 trains all epochs.")
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument("--ring_attn_size", type=int, default=1)
    parser.add_argument("--ring_head_stride", type=int, default=1)

    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument(
        "--lora_embedding_train_mode",
        type=str,
        default="embedding_lora",
        choices=["embedding_lora", "pause_token", "full", "auto", "off"],
    )

    parser.add_argument("--packing_samples", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--dataset_revision", type=str)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--multiturn", action="store_true", default=False)
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--output_key", type=str, default="answer")
    parser.add_argument("--prompt_type", type=str, default=None)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_pause_sft")
    parser.add_argument("--wandb_run_name", type=str, default="pause_sft_%s" % datetime.now().strftime("%m%dT%H:%M"))
    parser.add_argument("--use_tensorboard", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.multiturn:
        assert args.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    if args.input_template and "\\n" in args.input_template:
        print("[Warning] input_template contains literal \\n; decoding it.")
        args.input_template = args.input_template.encode().decode("unicode_escape")

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"
    if args.packing_samples and "flash_attention" not in args.attn_implementation:
        print("[Warning] Please use flash attention when --packing_samples is enabled.")
        args.attn_implementation = "flash_attention_2"

    train(args)
