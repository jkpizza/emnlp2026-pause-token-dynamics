import math
import os
import ray
from tqdm import tqdm
from transformers import AutoTokenizer
from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import (
    PauseGRPORemoteExperienceMaker,
    PauseGRPOSamplesGenerator,
)
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _aligned_rollout_sample_count(total_samples: int, args) -> int:
    """Keep whole GRPO groups while forcing equal actor-rank train steps."""
    n_samples_per_prompt = max(1, int(args.n_samples_per_prompt))
    micro_rollout_batch_size = max(1, int(args.micro_rollout_batch_size))
    micro_train_batch_size = max(1, int(args.micro_train_batch_size))
    effective_actor_num = (
        args.actor_num_nodes
        * args.actor_num_gpus_per_node
        // args.ring_attn_size
        // args.ds_tensor_parallel_size
    )
    if effective_actor_num <= 0:
        raise ValueError("effective actor count must be positive.")
    alignment_unit = math.lcm(
        n_samples_per_prompt,
        effective_actor_num * micro_rollout_batch_size,
        effective_actor_num * micro_train_batch_size,
    )
    return total_samples // alignment_unit * alignment_unit


@ray.remote
class PausePPOTrainer(BasePPOTrainer):
    """Single-policy PPO/GRPO trainer for pause-inserted rollouts."""

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )
        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(
                self.init_kl_coef, self.kl_target, self.kl_horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)
        tokenizer_path = self.args.pause_tokenizer_save_path or pretrain
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=not self.args.disable_fast_tokenizer,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.args.pause_token]}
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.samples_generator = PauseGRPOSamplesGenerator(
            self.vllm_engines, self.strategy, self.tokenizer, self.prompt_max_len
        )
        self.experience_maker = PauseGRPORemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
        )
        self.prepare_datasets()
        self._init_wandb()
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")

    def _trim_rollout_samples_for_alignment(self, rollout_samples):
        keep_count = _aligned_rollout_sample_count(len(rollout_samples), self.args)
        if keep_count == len(rollout_samples):
            return rollout_samples
        n_samples_per_prompt = max(1, int(self.args.n_samples_per_prompt))
        dropped = len(rollout_samples) - keep_count
        if keep_count == 0:
            logger.warning(
                "Pause GRPO rollout has %d sample(s), but none can be kept without breaking GRPO group and distributed microbatch alignment. Skip iteration.",
                len(rollout_samples),
            )
            return []
        logger.warning(
            "Trim pause GRPO rollout samples from %d to %d (drop %d sample(s), %d prompt group(s)) to keep actor ranks aligned.",
            len(rollout_samples),
            keep_count,
            dropped,
            dropped // n_samples_per_prompt,
        )
        return rollout_samples[:keep_count]

    def fit(self) -> None:
        args = self.args
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(
                self.actor_model_group.async_run_method(
                    method_name="get_checkpoint_states"
                )
            )[0]
            logger.info(f"checkpoint_states: {checkpoint_states}")
        else:
            checkpoint_states = {
                "global_step": 0,
                "episode": 0,
                "data_loader_state_dict": {},
            }
        self._broadcast_actor_to_vllm_for_rollout()
        steps = checkpoint_states["global_step"] + 1
        initial_step = steps
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)
        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                initial=steps,
            )
            filtered_samples = []
            number_of_samples = 0
            for (_, rand_prompts, labels) in self.prompts_dataloader:
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts, labels, global_step=steps, **self.generate_kwargs
                )
                pbar.update()
                if not rollout_samples:
                    logger.warning(
                        "Pause GRPO rollout batch is empty after invalid group filtering; skip iteration."
                    )
                    continue
                if any(
                    (
                        sample.rewards is None or sample.scores is None
                        for sample in rollout_samples
                    )
                ):
                    raise ValueError(
                        "PauseGRPO reward must be computed from original no-pause rollout text before experience making"
                    )
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    for i in range(
                        0, len(rollout_samples), self.args.n_samples_per_prompt
                    ):
                        batch_samples = rollout_samples[
                            i : i + self.args.n_samples_per_prompt
                        ]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue
                        avg_reward = sum(
                            (sample.scores[0].item() for sample in batch_samples)
                        ) / len(batch_samples)
                        (
                            min_reward,
                            max_reward,
                        ) = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-06 < avg_reward < max_reward - 1e-06:
                            filtered_samples.extend(batch_samples)
                    if (
                        len(filtered_samples) / self.args.n_samples_per_prompt
                        < self.args.rollout_batch_size
                    ):
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue
                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[
                        : self.args.rollout_batch_size * self.args.n_samples_per_prompt
                    ]
                    filtered_samples = []
                    number_of_samples = 0
                rollout_samples = self._trim_rollout_samples_for_alignment(
                    rollout_samples
                )
                if not rollout_samples:
                    continue
                experiences = self.experience_maker.make_experience_batch(
                    rollout_samples, global_step=steps
                )
                reward_status = self._reward_metrics(rollout_samples)
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)
                refs = self.actor_model_group.async_run_method_batch(
                    method_name="append", experience=experiences
                )
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(
                            method_name="append", experience=experiences
                        )
                    )
                ray.get(refs)
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=False
                )
                status = self.ppo_train(steps)
                if "kl" in status:
                    self.kl_ctl.update(status["kl"], len(rollout_samples))
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                status.update(reward_status)
                status = self._filter_training_status(status)
                logger.info(f"✨ Pause GRPO global step {steps}: {status}")
                status["generated_samples"] = [
                    sample0[0],
                    experiences[0].info["reward"][0],
                ]
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)
                steps = steps + 1
                if (
                    getattr(args, "max_updates", None) is not None
                    and steps - initial_step >= args.max_updates
                ):
                    if self._wandb is not None:
                        self._wandb.finish()
                    if self._tensorboard is not None:
                        self._tensorboard.close()
                    return
        if getattr(args, "max_updates", None) is not None and steps == initial_step:
            raise RuntimeError(
                "Smoke run completed no GRPO update; all rollout groups may have been invalid."
            )
        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
