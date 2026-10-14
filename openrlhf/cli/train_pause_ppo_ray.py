import argparse
import os
from datetime import datetime
import ray
from ray.util.placement_group import placement_group
from openrlhf.models.pause_lm import ensure_pause_tokenizer
from openrlhf.trainer.ppo_pause_trainer import PausePPOTrainer
from openrlhf.trainer.ppo_trainer import PPOTrainer
from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    PauseReferenceModelActor,
    RayActorGroup,
    ReferenceModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PausePolicyModelActor, PolicyModelActor
from openrlhf.utils import get_strategy


def log_ray_resources(tag):
    print(f"\n===== {tag} =====")
    print("cluster_resources:", ray.cluster_resources())
    print("available_resources:", ray.available_resources())


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _optional_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer or none, got {value!r}"
        ) from exc


def _build_ray_init_kwargs():
    env_vars = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
    }
    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "OPENRLHF_ACTOR_MASTER_PORT_BASE",
        "OPENRLHF_ACTOR_MASTER_PORT_STRIDE",
        "RAY_DEDUP_LOGS",
        "RAY_USE_RANDOM_PORTS",
    ):
        if key in os.environ:
            env_vars[key] = os.environ[key]
    init_kwargs = {"runtime_env": {"env_vars": env_vars}}
    ray_tmpdir = os.environ.get("RAY_TMPDIR") or os.environ.get("RAY_TEMP_DIR")
    if ray_tmpdir:
        init_kwargs["_temp_dir"] = os.path.abspath(ray_tmpdir)
    ray_node_ip = os.environ.get("RAY_NODE_IP_ADDRESS")
    if ray_node_ip:
        init_kwargs["_node_ip_address"] = ray_node_ip
    if "RAY_INCLUDE_DASHBOARD" in os.environ:
        init_kwargs["include_dashboard"] = _env_flag("RAY_INCLUDE_DASHBOARD")
    dashboard_port = _env_int("RAY_DASHBOARD_PORT")
    if dashboard_port is not None:
        init_kwargs["dashboard_port"] = dashboard_port
    metrics_port = _env_int("RAY_METRICS_EXPORT_PORT")
    if metrics_port is not None:
        init_kwargs["_metrics_export_port"] = metrics_port
    return init_kwargs


def train(args):
    from reproduction.grpo import prepare_data

    prepare_data(args)
    if not ray.is_initialized():
        ray_init_kwargs = _build_ray_init_kwargs()
        print(f"Initializing Ray with kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)
    use_pause_grpo = args.grpo_mode == "pause"
    if use_pause_grpo:
        if args.pause_tokenizer_save_path is None:
            args.pause_tokenizer_save_path = os.path.join(
                args.save_path, "pause_tokenizer"
            )
        os.makedirs(args.pause_tokenizer_save_path, exist_ok=True)
        ensure_pause_tokenizer(
            args.pretrain,
            args.pause_tokenizer_save_path,
            pause_token=args.pause_token,
            use_fast=not args.disable_fast_tokenizer,
            padding_side="left",
            print_fn=print,
        )
    strategy = get_strategy(args)
    strategy.print(args)
    actor_cls = PausePolicyModelActor if use_pause_grpo else PolicyModelActor
    reference_cls = PauseReferenceModelActor if use_pause_grpo else ReferenceModelActor
    trainer_cls = PausePPOTrainer if use_pause_grpo else PPOTrainer
    use_lora_disabled_ref_model = getattr(args, "use_lora_disabled_ref_model", False)
    log_ray_resources("before_pg")
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0 and (not use_lora_disabled_ref_model):
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), "actor and reference resources must match when colocated."
        bundles = [
            {"GPU": 1, "CPU": 1}
            for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)
        ]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        log_ray_resources("after_pg")
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = (
            args.max_len
            if args.max_len
            else args.prompt_max_len + args.generate_max_len
        )
        if args.colocate_all_models:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), "actor GPUs must equal vLLM engines * tensor parallel size when colocated."
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor

        vllm_tokenizer_path = args.pause_tokenizer_save_path if use_pause_grpo else None
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            pg if args.colocate_all_models else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
            LLMRayActor,
            "processed_logprobs"
            if not use_pause_grpo and args.enable_vllm_is_correction
            else None,
            tokenizer=vllm_tokenizer_path,
        )
    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        actor_cls,
        pg=pg,
        num_gpus_per_actor=0.15 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )
    log_ray_resources("after_actor_model")
    if args.init_kl_coef <= 0:
        ref_model = None
    elif use_lora_disabled_ref_model:
        ref_model = None
    else:
        ref_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            reference_cls,
            pg=pg,
            num_gpus_per_actor=0.15 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    log_ray_resources("after_ref_model")
    if not args.colocate_all_models:
        pg = None
    reward_model = None
    ppo_trainer = trainer_cls.remote(
        args.pretrain,
        strategy,
        actor_model,
        None,
        reward_model,
        ref_model,
        vllm_engines,
        prompt_split=args.prompt_split,
        eval_split=args.eval_split,
        do_sample=True,
        prompt_max_len=args.prompt_max_len,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len
        if args.max_len
        else args.prompt_max_len + args.generate_max_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())
    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(
        actor_model.async_init_model_from_pretrained(
            strategy, args.pretrain, max_steps, vllm_engines
        )
    )
    ray.get(refs)
    ray.get(ppo_trainer.fit.remote())
    ray.get(actor_model.async_save_model())
    if use_pause_grpo:
        tokenizer = ensure_pause_tokenizer(
            args.pretrain,
            args.pause_tokenizer_save_path,
            pause_token=args.pause_token,
            use_fast=not args.disable_fast_tokenizer,
            padding_side="left",
        )
        tokenizer.save_pretrained(args.save_path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_updates",
        type=int,
        default=None,
        help="Stop after this many rollout/update iterations (smoke validation).",
    )
    parser.add_argument(
        "--grpo_mode", type=str, default="pause", choices=["pause", "normal"]
    )
    parser.add_argument("--pause_token", type=str, default="<|PAUSE|>")
    parser.add_argument("--pause_tokens_per_step", type=int, default=1)
    parser.add_argument(
        "--pause_init_method",
        type=str,
        default="mean",
        choices=["mean", "gaussian_norm", "none"],
    )
    parser.add_argument("--pause_tokenizer_save_path", type=str, default=None)
    parser.add_argument(
        "--pause_reference_mode",
        type=str,
        default="original",
        choices=["original", "inserted"],
    )
    parser.add_argument(
        "--pause_posthoc_rollout_ratio_schedule", action="store_true", default=False
    )
    parser.add_argument("--pause_posthoc_rollout_start_ratio", type=float, default=1.0)
    parser.add_argument("--pause_posthoc_rollout_end_ratio", type=float, default=1.0)
    parser.add_argument(
        "--pause_posthoc_rollout_ratio_warmup_start_step", type=int, default=0
    )
    parser.add_argument(
        "--pause_posthoc_rollout_ratio_warmup_end_step", type=int, default=0
    )
    parser.add_argument(
        "--pause_allow_vllm_token_mismatch",
        action="store_true",
        default=False,
        help="Allow one-shot response_text tokenization even when it does not match vLLM completion.token_ids. Default False keeps the strict invariant that rollout visible ids are vLLM token ids.",
    )
    parser.add_argument(
        "--pause_prefix_decode_block_size",
        type=int,
        default=128,
        help="Chunk size for fallback mapping from vLLM token ids to text step boundaries. Larger values reduce checkpoint decode calls; smaller values make boundary search chunks narrower.",
    )
    parser.add_argument("--ref_num_nodes", type=int, default=1)
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--actor_num_nodes", type=int, default=1)
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--critic_num_nodes", type=int, default=1)
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--colocate_actor_ref", action="store_true", default=False)
    parser.add_argument("--colocate_critic_reward", action="store_true", default=False)
    parser.add_argument("--colocate_all_models", action="store_true", default=False)
    parser.add_argument("--vllm_num_engines", type=int, default=None)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm_enable_sleep", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument(
        "--enable_vllm_is_correction", action="store_true", default=False
    )
    parser.add_argument("--vllm_is_truncated_threshold", type=float, default=2)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/pause_grpo")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=100000000.0)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--enable_ema", action="store_true", default=False)
    parser.add_argument("--ema_beta", type=float, default=0.992)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant", action="store_true", default=False
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--deepspeed_enable_sleep", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--packing_samples", action="store_true", default=False)
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)
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
    parser.add_argument(
        "--use_lora_disabled_ref_model", action="store_true", default=False
    )
    parser.add_argument("--save_path", type=str, default="./ckpt/pause_grpo")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--vllm_generate_batch_size", type=int, default=None)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None)
    parser.add_argument("--dual_clip", type=float, default=None)
    parser.add_argument("--value_clip", type=float, default=0.5)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1)
    parser.add_argument("--n_samples_per_prompt", type=int, default=8)
    parser.add_argument("--save_value_network", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-06)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-06)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument(
        "--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"]
    )
    parser.add_argument(
        "--kl_estimator", type=str, default="k3", choices=["k1", "k2", "k3"]
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--entropy_loss_coef", type=float, default=None)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10))
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="group_norm",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False)
    parser.add_argument("--no_advantage_std_norm", action="store_true", default=False)
    parser.add_argument("--overlong_buffer_len", type=float, default=None)
    parser.add_argument("--overlong_penalty_factor", type=float, default=1)
    parser.add_argument("--pause_clip_penalty", type=float, default=0.0)
    parser.add_argument("--pause_length_target", type=int, default=0)
    parser.add_argument("--pause_over_target_token_penalty", type=float, default=0.0)
    parser.add_argument("--pause_penalty_warmup_steps", type=int, default=0)
    parser.add_argument("--pause_penalty_ramp_steps", type=int, default=0)
    parser.add_argument(
        "--pause_penalty_correct_only", action="store_true", default=False
    )
    parser.add_argument("--ring_attn_size", type=int, default=1)
    parser.add_argument("--ring_head_stride", type=int, default=1)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--critic_pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument("--prompt_data_config", type=str, default=None)
    parser.add_argument("--prompt_data_revision", type=str, default=None)
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_dataset_name", type=str, default=None)
    parser.add_argument("--eval_dataset2", type=str, default=None)
    parser.add_argument("--eval_dataset2_name", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--eval_split2", type=str, default=None)
    parser.add_argument("--eval_input_key", type=str, default="text")
    parser.add_argument("--eval_label_key", type=str, default="label")
    parser.add_argument("--eval_input_key2", type=str, default="text")
    parser.add_argument("--eval_label_key2", type=str, default="label")
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=1)
    parser.add_argument("--eval_pass_at_k", type=_optional_int, default=4)
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--answer_trigger", type=str, default="The answer is:")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_pause_grpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="pause_grpo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--use_tensorboard", type=str, default=None)
    parser.add_argument("--dynamic_filtering", action="store_true", default=False)
    parser.add_argument(
        "--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float
    )
    parser.add_argument("--perf", action="store_true", default=False)
    parser.add_argument("--use_ms", action="store_true", default=False)
    parser.add_argument("--debug_skip_device_mesh", action="store_true", default=False)
    return parser


def validate_args(args):
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be positive.")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("--max_updates must be positive.")
    args.use_named_eval_logging = True
    if args.eval_pass_at_k is None:
        args.eval_n_samples_per_prompt = 1
    else:
        if args.eval_pass_at_k <= 1:
            raise ValueError("--eval_pass_at_k must be greater than 1 or none.")
        args.eval_n_samples_per_prompt = args.eval_pass_at_k
    if args.eval_split2 is None:
        args.eval_split2 = args.eval_split
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)
    if args.pretrain is None:
        raise ValueError("--pretrain must be set.")
    if args.prompt_data is None:
        raise ValueError("--prompt_data must be set.")
    if not args.input_key or not args.label_key:
        raise ValueError(
            "GRPO requires --input_key and --label_key for questions and final answers."
        )
    if args.vllm_num_engines is None or args.vllm_num_engines <= 0:
        raise ValueError("--vllm_num_engines > 0 is required.")
    if (
        args.advantage_estimator
        in ["rloo", "reinforce_baseline", "group_norm", "dr_grpo"]
        and args.n_samples_per_prompt <= 1
    ):
        raise ValueError(
            f"{args.advantage_estimator} requires --n_samples_per_prompt > 1."
        )
    if args.use_lora_disabled_ref_model and args.lora_rank <= 0:
        raise ValueError("--use_lora_disabled_ref_model requires --lora_rank > 0.")
    if not 0.0 <= args.pause_posthoc_rollout_start_ratio <= 1.0:
        raise ValueError("--pause_posthoc_rollout_start_ratio must be in [0, 1].")
    if not 0.0 <= args.pause_posthoc_rollout_end_ratio <= 1.0:
        raise ValueError("--pause_posthoc_rollout_end_ratio must be in [0, 1].")
    if args.pause_posthoc_rollout_ratio_warmup_start_step < 0:
        raise ValueError(
            "--pause_posthoc_rollout_ratio_warmup_start_step must be non-negative."
        )
    if args.pause_posthoc_rollout_ratio_warmup_end_step < 0:
        raise ValueError(
            "--pause_posthoc_rollout_ratio_warmup_end_step must be non-negative."
        )
    if (
        args.pause_posthoc_rollout_ratio_warmup_end_step
        < args.pause_posthoc_rollout_ratio_warmup_start_step
    ):
        raise ValueError(
            "--pause_posthoc_rollout_ratio_warmup_end_step must be greater than or equal to --pause_posthoc_rollout_ratio_warmup_start_step."
        )
    if args.grpo_mode == "pause" and args.pause_tokenizer_save_path is None:
        args.pause_tokenizer_save_path = os.path.join(args.save_path, "pause_tokenizer")
    if args.grpo_mode == "pause":
        if args.use_lora_disabled_ref_model and args.lora_embedding_train_mode not in [
            "embedding_lora",
            "off",
        ]:
            raise ValueError(
                "--use_lora_disabled_ref_model with pause GRPO requires --lora_embedding_train_mode embedding_lora or off."
            )
        if args.pause_tokenizer_save_path is None:
            args.pause_tokenizer_save_path = os.path.join(
                args.save_path, "pause_tokenizer"
            )
        if args.disable_fast_tokenizer:
            raise ValueError(
                "Pause GRPO step-token alignment requires a fast tokenizer for return_offsets_mapping. Remove --disable_fast_tokenizer."
            )
        if args.packing_samples and "flash_attention" not in args.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.attn_implementation = "flash_attention_2"
        if args.use_dynamic_batch:
            raise ValueError(
                "Pause GRPO v1 does not support --use_dynamic_batch because it implies packing."
            )
        if args.ring_attn_size > 1:
            raise ValueError("Pause GRPO v1 does not support --ring_attn_size > 1.")
        if args.enable_vllm_is_correction:
            raise ValueError(
                "Pause GRPO v1 does not support --enable_vllm_is_correction."
            )
        if args.policy_loss_type != "ppo":
            raise ValueError(
                "Pause GRPO currently supports --policy_loss_type ppo only."
            )
    else:
        if args.pause_posthoc_rollout_ratio_schedule:
            raise ValueError(
                "--pause_posthoc_rollout_ratio_schedule is only valid with --grpo_mode pause."
            )
        if args.ring_attn_size > 1 and (not args.packing_samples):
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.packing_samples = True
        if args.use_dynamic_batch:
            if not args.packing_samples:
                print(
                    "[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled."
                )
                args.packing_samples = True
            if args.rollout_max_tokens_per_gpu is None:
                print(
                    "[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu."
                )
                args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu
        if args.packing_samples and "flash_attention" not in args.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.attn_implementation = "flash_attention_2"
    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )
    if args.vllm_enable_sleep and (not args.colocate_all_models):
        print(
            "Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled."
        )
        args.vllm_enable_sleep = False
    if args.use_kl_loss and args.kl_estimator not in ["k2", "k3"]:
        print(
            f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss"
        )
    if args.grpo_mode == "normal" and args.dynamic_filtering:
        if (
            args.dynamic_filtering_reward_range[0]
            >= args.dynamic_filtering_reward_range[1]
        ):
            raise ValueError(
                "dynamic_filtering_reward_range[0] must be less than dynamic_filtering_reward_range[1]."
            )
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size
    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        patch_hub()
    return args


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    train(validate_args(parsed_args))
