from openrlhf.trainer.ppo_utils.grpo_metrics import masked_mean_safe
import math
import os
import socket
from abc import ABC
from typing import Dict, List, Optional, Union
import deepspeed
import ray
import torch
import torch.distributed
from peft.tuners.lora import LoraLayer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler
from openrlhf.models import Actor, PauseLM, PolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ppo_utils.grpo_metrics import should_skip_pause_training_metric
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from openrlhf.utils.distributed_util import (
    stateless_init_process_group,
    torch_dist_barrier_and_cuda_sync,
)
from openrlhf.utils.logging_utils import init_logger
from ..ppo_utils import NaiveReplayBuffer
import json

logger = init_logger(__name__)
from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta
        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs
        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=self.args.vllm_is_truncated_threshold
            if self.args.enable_vllm_is_correction
            else None,
        )
        self.aux_loss = self.args.aux_loss_coef > 1e-08
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.args.colocate_all_models:
            self.use_cuda_ipc = True
        if (
            self.vllm_engines is not None
            and (not self.use_cuda_ipc)
            and (torch.distributed.get_rank() == 0)
        ):
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            (vllm_num_engines, vllm_tensor_parallel_size) = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for (i, engine) in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(
                    world_size=world_size,
                    rank=0,
                    backend=backend,
                    group_name=group_name,
                )
                self._model_update_group = group_name
            else:
                self._model_update_group = stateless_init_process_group(
                    master_address,
                    master_port,
                    0,
                    world_size,
                    torch.cuda.current_device(),
                )
            ray.get(refs)
        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, kl_ctl: float, global_step: Optional[int] = None):
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)
        buffer_len = len(self.replay_buffer)
        log_all_ranks = (
            os.environ.get("OPENRLHF_LOG_ALL_RANK_REPLAY_BUFFER", "0") == "1"
        )
        if log_all_ranks or self.strategy.is_rank_0():
            sample_bs = self.replay_buffer.sample_batch_size
            local_num_batches = buffer_len // sample_bs if sample_bs > 0 else 0
            logger.info(
                f"[ActorPPOTrainer rank={self.strategy.get_rank()}] replay_buffer size={buffer_len}, sample_batch_size={sample_bs}, num_batches(floor)={local_num_batches}, micro_train_batch_size={self.micro_train_batch_size}"
            )
        if buffer_len == 0:
            logger.warning("[ActorPPOTrainer] replay_buffer is empty. Skip training.")
            return {}
        not_shuffle = (
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()
        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for (step, experience) in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step)
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]
                short_status = {
                    "act_loss": status["policy_loss"],
                    "reward": status["reward"],
                    "return": status["return"],
                    "gen_len": status["response_length"],
                    "tot_len": status["total_length"],
                    "kl": status["kl"],
                    "act_lr": status["actor_lr"],
                }
                optional_status_map = {"entropy_loss": "ent_loss"}
                for (status_key, short_key) in optional_status_map.items():
                    if status_key in status:
                        short_status[short_key] = status[status_key]
                status_list.append(status)
                pbar.set_postfix(short_status)
        if not status_list:
            logger.warning(
                "[ActorPPOTrainer] No optimization step was executed. Dataloader was empty after batching."
            )
            return {}
        status_mean = status_list[0]
        for m in status_list[1:]:
            for (k, v) in m.items():
                status_mean[k] += v
        for k in status_mean.keys():
            status_mean[k] /= len(status_list)
        tracked_keys = (
            "reward",
            "raw_reward",
            "r_disc_raw",
            "rloo_adv",
            "normalized_adv",
        )
        for key in tracked_keys:
            vals = [float(m[key]) for m in status_list if key in m]
            if not vals:
                continue
            t = torch.tensor(vals, dtype=torch.float32)
            status_mean[f"{key}_last"] = t[-1].item()
            status_mean[f"{key}_std"] = (
                t.std(unbiased=False).item() if t.numel() > 1 else 0.0
            )
            status_mean[f"{key}_abs_mean"] = t.abs().mean().item()
        return status_mean

    def training_step(
        self,
        experience: Experience,
        kl_ctl: float,
        step: int,
        global_step: Optional[int] = None,
    ) -> Dict[str, float]:
        self.actor.train()
        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs
        (action_log_probs, output) = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.entropy_loss_coef is not None,
        )
        (actor_loss, clip_ratio, ppo_kl, vllm_kl) = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()
        if self.args.use_kl_loss:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(
                    action_log_probs,
                    dtype=action_log_probs.dtype,
                    device=action_log_probs.device,
                )
            kl_mean = masked_mean(kl, experience.action_mask)
            kl_loss = kl_mean * kl_ctl
            experience.info["kl"] = kl_mean.detach()
        else:
            kl_loss = 0
        loss = actor_loss + kl_loss
        if self.aux_loss:
            loss += output.aux_loss * self.args.aux_loss_coef
        entropy_loss = None
        if self.args.entropy_loss_coef is not None:
            entropy_window = output.entropy[:, -experience.action_mask.shape[1] :]
            entropy_mask = experience.action_mask
            entropy_loss = masked_mean_safe(entropy_window, entropy_mask)
            if self.args.entropy_loss_coef != 0:
                text_entropy_bonus = entropy_loss * self.args.entropy_loss_coef
                loss -= text_entropy_bonus
        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]
        self.strategy.backward(loss, self.actor, self.actor_optim)
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(
                    self.actor_optim, self.actor, self.actor_scheduler, name="actor"
                )
        else:
            self.strategy.optimizer_step(
                self.actor_optim, self.actor, self.actor_scheduler, name="actor"
            )
        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(
                        self.actor, self.ema_model, self.ema_beta, "cuda"
                    )
            else:
                self.strategy.moving_average(
                    self.actor, self.ema_model, self.ema_beta, "cuda"
                )
        status = {
            "policy_loss": actor_loss.detach().item(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
        }
        if self.args.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()
        for (k, v) in experience.info.items():
            if should_skip_pause_training_metric(k):
                continue
            if isinstance(v, list):
                if all((isinstance(x, (int, float)) for x in v)):
                    status[k] = torch.tensor(v, dtype=torch.float).mean().item()
                elif all((isinstance(x, torch.Tensor) for x in v)):
                    flat = torch.cat([x.float().reshape(-1).detach().cpu() for x in v])
                    if flat.numel() == len(v):
                        status[k] = flat.mean().item()
            elif isinstance(v, torch.Tensor):
                if k not in {"boundary_mask", "pause_ref_boundary_mask"}:
                    status[k] = v.float().mean().item()
        return status

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())
        torch.cuda.empty_cache()
        model = self.actor.model.module

        def _is_lora_param(name: str) -> bool:
            lora_param_markers = (
                ".lora_A.",
                ".lora_B.",
                ".lora_embedding_A.",
                ".lora_embedding_B.",
                ".lora_magnitude_vector.",
            )
            return any((marker in name for marker in lora_param_markers))

        def _peft_name_to_vllm_name(name: str) -> Optional[str]:
            if _is_lora_param(name):
                return None
            if ".original_module." in name:
                return None
            if ".modules_to_save." in name:
                (module_name, _, saved_name) = name.partition(".modules_to_save.")
                saved_parts = saved_name.split(".", 1)
                if len(saved_parts) != 2:
                    return None
                name = f"{module_name}.{saved_parts[1]}"
            for prefix in ("base_model.model.", "base_model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            name = name.replace(".base_layer.", ".")
            return name

        def _active_lora_adapters(module: LoraLayer) -> List[str]:
            adapters = getattr(module, "active_adapters", None)
            if adapters is None:
                adapter = getattr(module, "active_adapter", None)
                if adapter is None:
                    return []
                adapters = [adapter] if isinstance(adapter, str) else list(adapter)
            elif isinstance(adapters, str):
                adapters = [adapters]
            return [
                adapter for adapter in adapters if adapter and adapter != "__base__"
            ]

        def _lora_delta_params(
            module: LoraLayer, base_param: torch.nn.Parameter
        ) -> List[torch.nn.Parameter]:
            params = [base_param]
            for adapter in _active_lora_adapters(module):
                if hasattr(module, "lora_A") and adapter in module.lora_A:
                    params.extend(module.lora_A[adapter].parameters())
                if hasattr(module, "lora_B") and adapter in module.lora_B:
                    params.extend(module.lora_B[adapter].parameters())
                if (
                    hasattr(module, "lora_embedding_A")
                    and adapter in module.lora_embedding_A
                ):
                    params.append(module.lora_embedding_A[adapter])
                if (
                    hasattr(module, "lora_embedding_B")
                    and adapter in module.lora_embedding_B
                ):
                    params.append(module.lora_embedding_B[adapter])
            return params

        def _effective_lora_weight(
            param: torch.nn.Parameter, module: Optional[LoraLayer]
        ) -> torch.Tensor:
            weight = param.data
            if not isinstance(module, LoraLayer) or not hasattr(
                module, "get_delta_weight"
            ):
                return weight
            if getattr(module, "_disable_adapters", False):
                return weight
            effective_weight = weight
            merged_adapters = set(getattr(module, "merged_adapters", []) or [])
            for adapter in _active_lora_adapters(module):
                if adapter in merged_adapters:
                    continue
                try:
                    delta = module.get_delta_weight(adapter)
                except KeyError:
                    continue
                delta = delta.to(device=weight.device, dtype=weight.dtype)
                effective_weight = effective_weight + delta
            return effective_weight

        def _iter_vllm_sync_items():
            is_peft_model = hasattr(model, "peft_config") and hasattr(
                model, "base_model"
            )
            if not is_peft_model:
                for (param_name, param) in model.named_parameters():
                    yield (param_name, param_name, param, None, [param])
                return
            module_map = dict(model.named_modules())
            if torch.distributed.get_rank() == 0:
                logger.info("Broadcasting PEFT LoRA effective weights to vLLM.")
            for (param_name, param) in model.named_parameters():
                vllm_name = _peft_name_to_vllm_name(param_name)
                if vllm_name is None:
                    continue
                if any(
                    (
                        marker in vllm_name
                        for marker in ("base_model", ".lora_", ".base_layer.")
                    )
                ):
                    raise ValueError(
                        f"Unexpected PEFT wrapper component in vLLM weight name: {param_name} -> {vllm_name}"
                    )
                lora_module = None
                gather_params = [param]
                if param_name.endswith(".base_layer.weight"):
                    module_name = param_name[: -len(".base_layer.weight")]
                    module = module_map.get(module_name)
                    if isinstance(module, LoraLayer):
                        lora_module = module
                        gather_params = _lora_delta_params(module, param)
                yield (param_name, vllm_name, param, lora_module, gather_params)

        sync_items = list(_iter_vllm_sync_items())
        (count, num_params) = (0, len(sync_items))

        def _broadcast_tensor(name, tensor, empty_cache):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            if torch.distributed.get_rank() == 0:
                tensor = tensor.detach().contiguous()
                refs = [
                    engine.update_weight.remote(
                        name,
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        empty_cache=empty_cache,
                    )
                    for engine in self.vllm_engines
                ]
                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(tensor, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(
                        tensor, src=0, stream=torch.cuda.current_stream()
                    )
                ray.get(refs)

        def _handle_cuda_ipc(name, tensor, empty_cache):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = tensor.detach().contiguous().clone()
            ipc_handle = reduce_tensor(weight)
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)
            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=weight.dtype,
                        shape=weight.shape,
                        ipc_handles=ipc_handles,
                        empty_cache=empty_cache,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            del weight
            torch_dist_barrier_and_cuda_sync()

        for (_, vllm_name, param, lora_module, gather_params) in sync_items:
            count += 1
            if not self.use_cuda_ipc:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams(
                        gather_params, model, enabled=True
                    ):
                        tensor = _effective_lora_weight(param, lora_module)
                        _broadcast_tensor(vllm_name, tensor, count == num_params)
                else:
                    with deepspeed.zero.GatheredParameters(
                        gather_params, enabled=self.strategy.args.zero_stage == 3
                    ):
                        tensor = _effective_lora_weight(param, lora_module)
                        _broadcast_tensor(vllm_name, tensor, count == num_params)
            elif self.strategy.args.ds_tensor_parallel_size > 1:
                with deepspeed.module_inject.layers.GatherReplacedLayerParams(
                    gather_params, model, enabled=True
                ):
                    tensor = _effective_lora_weight(param, lora_module)
                    _handle_cuda_ipc(vllm_name, tensor, count == num_params)
            else:
                with deepspeed.zero.GatheredParameters(
                    gather_params, enabled=self.strategy.args.zero_stage == 3
                ):
                    tensor = _effective_lora_weight(param, lora_module)
                    _handle_cuda_ipc(vllm_name, tensor, count == num_params)
        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None
    ):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps
        if getattr(args, "vllm_num_engines", 0) > 0:
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
        self._setup_distributed(strategy)
        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)
        ds_config = strategy.get_ds_train_config(is_actor=True)
        logger.info(f"ds_config = {json.dumps(ds_config, indent=2, sort_keys=True)}")
        self.tokenizer = get_tokenizer(
            pretrain,
            actor.model,
            "left",
            strategy,
            use_fast=not strategy.args.disable_fast_tokenizer,
        )
        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.attn_implementation,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None
        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.actor_learning_rate,
            betas=strategy.args.adam_betas,
            weight_decay=args.l2,
        )
        logger.info(f"optimizer class = {actor_optim.__class__}")
        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )
        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )
        (self.actor, self.actor_optim, self.actor_scheduler) = strategy.prepare(
            (actor, actor_optim, actor_scheduler), is_rlhf=True
        )
        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            (_, states) = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states[
                "data_loader_state_dict"
            ]
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0, global_step: Optional[int] = None):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl, global_step=global_step)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def _forward_action_log_probs(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.actor(
            sequences,
            action_mask,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self._forward_action_log_probs(
                sequences.to(device), action_mask.to(device), attention_mask.to(device)
            )
        self.actor.train()
        return action_log_probs.to("cpu")

    def forward_with_lora_disabled_ref(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ref_sequences: Optional[torch.LongTensor] = None,
        ref_action_mask: Optional[Union[int, list[int]]] = None,
        ref_attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        ref_packed_seq_lens=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.cuda.current_device()
        peft_model = self.strategy._unwrap_model(self.actor)
        if not hasattr(peft_model, "disable_adapter"):
            raise ValueError(
                "--use_lora_disabled_ref_model requires the actor to be a PEFT LoRA model."
            )
        self.actor.eval()
        try:
            with torch.no_grad():
                action_log_probs = self._forward_action_log_probs(
                    sequences.to(device),
                    action_mask.to(device),
                    attention_mask.to(device),
                ).to("cpu")
                ref_sequences = sequences if ref_sequences is None else ref_sequences
                ref_action_mask = (
                    action_mask if ref_action_mask is None else ref_action_mask
                )
                ref_attention_mask = (
                    attention_mask if ref_attention_mask is None else ref_attention_mask
                )
                with peft_model.disable_adapter():
                    ref_action_log_probs = self._forward_action_log_probs(
                        ref_sequences.to(device),
                        ref_action_mask.to(device),
                        ref_attention_mask.to(device),
                    ).to("cpu")
        finally:
            self.actor.train()
        return (action_log_probs, ref_action_log_probs)

    def broadcast_to_vllm(self):
        self.trainer._broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PausePolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None
    ):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps
        if (
            getattr(args, "vllm_num_engines", 0) > 0
            and getattr(args, "vllm_sync_backend", "nccl") == "nccl"
        ):
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
        self._setup_distributed(strategy)
        actor = PauseLM(
            pretrain,
            strategy=strategy,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            lora_embedding_train_mode=strategy.args.lora_embedding_train_mode,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            pause_token=strategy.args.pause_token,
            pause_init_method=strategy.args.pause_init_method,
            pause_tokenizer_save_path=strategy.args.pause_tokenizer_save_path,
            use_fast=not strategy.args.disable_fast_tokenizer,
            padding_side="left",
        )
        strategy.print(actor)
        ds_config = strategy.get_ds_train_config(is_actor=True)
        logger.info(f"ds_config = {json.dumps(ds_config, indent=2, sort_keys=True)}")
        self.tokenizer = actor.get_tokenizer()
        if args.enable_ema:
            ema_model = PauseLM(
                pretrain,
                strategy=strategy,
                attn_implementation=strategy.args.attn_implementation,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
                temperature=strategy.args.temperature,
                use_liger_kernel=strategy.args.use_liger_kernel,
                pause_token=strategy.args.pause_token,
                pause_init_method=strategy.args.pause_init_method,
                pause_tokenizer_save_path=strategy.args.pause_tokenizer_save_path,
                use_fast=not strategy.args.disable_fast_tokenizer,
                padding_side="left",
            )
        else:
            ema_model = None
        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.actor_learning_rate,
            betas=strategy.args.adam_betas,
            weight_decay=args.l2,
        )
        logger.info(f"optimizer class = {actor_optim.__class__}")
        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )
        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )
        (self.actor, self.actor_optim, self.actor_scheduler) = strategy.prepare(
            (actor, actor_optim, actor_scheduler), is_rlhf=True
        )
        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            (_, states) = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states[
                "data_loader_state_dict"
            ]
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0, global_step: Optional[int] = None):
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl, global_step=global_step)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def _forward_action_log_probs(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        return self.actor(
            sequences,
            action_mask,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self._forward_action_log_probs(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                packed_seq_lens=packed_seq_lens,
            )
        self.actor.train()
        return action_log_probs.to("cpu")

    def forward_with_lora_disabled_ref(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ref_sequences: Optional[torch.LongTensor] = None,
        ref_action_mask: Optional[Union[int, list[int]]] = None,
        ref_attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        ref_packed_seq_lens=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.cuda.current_device()
        peft_model = self.strategy._unwrap_model(self.actor)
        if not hasattr(peft_model, "disable_adapter"):
            raise ValueError(
                "--use_lora_disabled_ref_model requires the actor to be a PEFT LoRA model."
            )
        self.actor.eval()
        try:
            with torch.no_grad():
                action_log_probs = self._forward_action_log_probs(
                    sequences.to(device),
                    action_mask.to(device),
                    attention_mask.to(device),
                    packed_seq_lens=packed_seq_lens,
                ).to("cpu")
                ref_sequences = sequences if ref_sequences is None else ref_sequences
                ref_action_mask = (
                    action_mask if ref_action_mask is None else ref_action_mask
                )
                ref_attention_mask = (
                    attention_mask if ref_attention_mask is None else ref_attention_mask
                )
                with peft_model.disable_adapter():
                    ref_action_log_probs = self._forward_action_log_probs(
                        ref_sequences.to(device),
                        ref_action_mask.to(device),
                        ref_attention_mask.to(device),
                        packed_seq_lens=ref_packed_seq_lens,
                    ).to("cpu")
        finally:
            self.actor.train()
        return (action_log_probs, ref_action_log_probs)

    def broadcast_to_vllm(self):
        self.trainer._broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        if not args.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        torch_dist_barrier_and_cuda_sync()
