import os
import re
import time
from abc import ABC
from datetime import timedelta
import ray
import torch
from tqdm import tqdm
from openrlhf.datasets import PromptDataset
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.grpo_metrics import (
    reward_group_metrics,
    rewards_from_samples,
    should_skip_pause_training_metric,
)
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


def _infer_eval_dataset_name(dataset_path: str, fallback: str = "eval") -> str:
    if not dataset_path:
        return fallback
    first_dataset = dataset_path.split(",")[0].strip().split("@")[0].strip()
    basename = os.path.basename(first_dataset.rstrip("/")) or first_dataset.replace(
        "/", "_"
    )
    (stem, ext) = os.path.splitext(basename)
    if ext.lower() in {".json", ".jsonl", ".csv", ".parquet", ".arrow"}:
        basename = stem
    name = re.sub("[^0-9A-Za-z_.-]+", "_", basename).strip("_")
    return name or fallback


class BasePPOTrainer(ABC):
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
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = get_tokenizer(
            pretrain,
            None,
            "left",
            strategy,
            use_fast=not self.args.disable_fast_tokenizer,
        )
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines
        self.prompt_split = prompt_split
        self.eval_split = eval_split
        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs
        self.max_epochs = self.args.max_epochs
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon
        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.eval_dataloaders = None
        self.max_steps = None
        self.samples_generator = None
        self.experience_maker = None
        from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        os.environ.setdefault("WANDB_MODE", "offline")
        self._defined_wandb_eval_metrics = set()
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if os.environ.get("WANDB_MODE", "offline") not in {
                "offline",
                "disabled",
            } and (not wandb.api.api_key):
                wandb.login()
            wandb_dir = os.path.join(os.path.dirname(self.args.save_path), "logs")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                dir=wandb_dir,
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step", step_sync=True)
            wandb.define_metric("eval/*", step_metric="global_step", step_sync=True)
            self.generated_samples_table = wandb.Table(
                columns=["global_step", "text", "reward"]
            )
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _filter_training_status(self, status):
        return {
            key: value
            for (key, value) in status.items()
            if not should_skip_pause_training_metric(key)
        }

    @staticmethod
    def _wandb_eval_metric_aliases(logs):
        aliased_logs = dict(logs)
        for (key, value) in logs.items():
            if "pass@" not in key:
                continue
            aliased_logs.setdefault(key.replace("pass@", "pass_at_"), value)
            aliased_logs.setdefault(key.replace("pass@", "pass"), value)
        return aliased_logs

    def _wandb_log_at_global_step(self, logs, global_step):
        if self._wandb is None or not logs:
            return
        wandb_logs = dict(logs)
        wandb_logs["global_step"] = global_step
        for metric_key in wandb_logs:
            if (
                metric_key.startswith("eval/")
                and metric_key not in self._defined_wandb_eval_metrics
            ):
                self._wandb.define_metric(
                    metric_key, step_metric="global_step", step_sync=True
                )
                self._defined_wandb_eval_metrics.add(metric_key)
        self._wandb.log(wandb_logs, step=global_step)

    def _broadcast_actor_to_vllm_for_rollout(self):
        if self.vllm_engines is None:
            return
        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(
                self.actor_model_group.async_run_method(method_name="reload_states")
            )
        self._broadcast_to_vllm()
        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(
                self.actor_model_group.async_run_method(method_name="offload_states")
            )

    def _reward_metrics(self, rollout_samples):
        return reward_group_metrics(
            rewards_from_samples(rollout_samples), self.args.n_samples_per_prompt
        )

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def ppo_train(self, global_steps):
        status = {}
        if self.critic_model_group is not None:
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(
                    self.critic_model_group.async_run_method(
                        method_name="reload_states"
                    )
                )
            critic_status_ref = self.critic_model_group.async_run_method(
                method_name="fit"
            )
            if (
                self.strategy.args.colocate_all_models
                or self.strategy.args.deepspeed_enable_sleep
            ):
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(
                    self.critic_model_group.async_run_method(
                        method_name="offload_states"
                    )
                )
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(
                    self.actor_model_group.async_run_method(method_name="reload_states")
                )
            actor_status_ref = self.actor_model_group.async_run_method(
                method_name="fit", kl_ctl=self.kl_ctl.value, global_step=global_steps
            )
            status.update(ray.get(actor_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(
                    self.actor_model_group.async_run_method(
                        method_name="offload_states"
                    )
                )
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()
        if self.critic_model_group and (not self.strategy.args.colocate_all_models):
            status.update(ray.get(critic_status_ref)[0])
        return status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        ray.get(
            self.actor_model_group.async_run_method(method_name="broadcast_to_vllm")
        )
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(
        self, args, global_step, step_bar, logs_dict={}, client_states={}
    ):
        logs_dict = self._filter_training_status(logs_dict)
        wandb_payload = {}
        if global_step % args.logging_steps == 0:
            if self._wandb is not None:
                train_logs_dict = dict(logs_dict)
                generated_samples = train_logs_dict.pop("generated_samples", None)
                if generated_samples is not None:
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns,
                        data=self.generated_samples_table.data,
                    )
                    new_table.add_data(global_step, *generated_samples)
                    self.generated_samples_table = new_table
                    wandb_payload["train/generated_samples"] = new_table
                wandb_payload.update(
                    {
                        f"train/{k}": v
                        for (k, v) in train_logs_dict.items()
                        if k != "global_step"
                    }
                )
            elif self._tensorboard is not None:
                for (k, v) in logs_dict.items():
                    if k == "generated_samples":
                        (text, reward) = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text(
                            "train/generated_samples", formatted_text, global_step
                        )
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)
        if global_step % args.eval_steps == 0:
            if getattr(args, "use_named_eval_logging", False):
                for eval_entry in self.eval_dataloaders or []:
                    eval_dataloader = eval_entry["dataloader"]
                    if eval_dataloader and len(eval_dataloader) > 0:
                        eval_result = self.evaluate(
                            eval_dataloader,
                            global_step,
                            0.0,
                            1,
                            eval_name=eval_entry["name"],
                            pass_at_k=None,
                            log_pass_at_1=True,
                            log_pass_at_k=False,
                            log_to_wandb=False,
                            log_to_tensorboard=self._wandb is None,
                        )
                        if self._wandb is not None:
                            wandb_payload.update(eval_result["wandb_logs"])
                        eval_pass_at_k = getattr(args, "eval_pass_at_k", None)
                        if eval_pass_at_k is not None:
                            eval_result = self.evaluate(
                                eval_dataloader,
                                global_step,
                                args.eval_temperature,
                                eval_pass_at_k,
                                eval_name=eval_entry["name"],
                                pass_at_k=eval_pass_at_k,
                                log_pass_at_1=False,
                                log_pass_at_k=True,
                                log_to_wandb=False,
                                log_to_tensorboard=self._wandb is None,
                            )
                            if self._wandb is not None:
                                wandb_payload.update(eval_result["wandb_logs"])
            elif self.eval_dataloader and len(self.eval_dataloader) > 0:
                eval_result = self.evaluate(
                    self.eval_dataloader,
                    global_step,
                    args.eval_temperature,
                    args.eval_n_samples_per_prompt,
                    log_to_wandb=False,
                    log_to_tensorboard=self._wandb is None,
                )
                if self._wandb is not None:
                    wandb_payload.update(eval_result["wandb_logs"])
        self._wandb_log_at_global_step(wandb_payload, global_step)
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", tag=tag
                    )
                )
            ray.get(ref)

    def evaluate(
        self,
        eval_dataloader,
        global_step,
        temperature=0.6,
        n_samples_per_prompt=1,
        eval_name=None,
        pass_at_k=None,
        log_pass_at_1=True,
        log_pass_at_k=True,
        log_to_wandb=True,
        log_to_tensorboard=True,
    ):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt
            eval_name: Optional metric namespace for named eval datasets
            pass_at_k: Optional pass@k value to log for named eval datasets
            log_pass_at_1: Whether to log pass@1 from this evaluation pass
            log_pass_at_k: Whether to log pass@k from this evaluation pass
        """
        start_time = time.time()
        eval_label = f" ({eval_name})" if eval_name else ""
        logger.info(
            f"⏰ Evaluation{eval_label} start time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        pass_at_k_value = (
            int(pass_at_k) if pass_at_k is not None and int(pass_at_k) > 1 else None
        )
        if pass_at_k_value is not None and n_samples_per_prompt < pass_at_k_value:
            raise ValueError(
                f"pass@{pass_at_k_value} requires n_samples_per_prompt >= {pass_at_k_value}, got {n_samples_per_prompt}."
            )
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        with torch.no_grad():
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}
            for (datasources, prompts, labels) in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                for (prompt, datasource) in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts, all_labels, global_step=global_step, **generate_kwargs
            )
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)
            rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)
            if eval_name:
                global_metrics = {"count": 0}
                if log_pass_at_1:
                    global_metrics["pass1"] = 0
                if log_pass_at_k and pass_at_k_value is not None:
                    global_metrics[f"pass{pass_at_k_value}"] = 0
            else:
                global_metrics = {}
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                original_prompt = all_prompts[i * n_samples_per_prompt]
                if eval_name:
                    metrics = global_metrics
                else:
                    datasource = prompt_to_datasource[original_prompt]
                    if datasource not in global_metrics:
                        global_metrics[datasource] = {
                            f"pass{n_samples_per_prompt}": 0,
                            "pass1": 0,
                            "count": 0,
                        }
                    metrics = global_metrics[datasource]
                chunk_rewards = rewards[i]
                if eval_name:
                    if log_pass_at_k and pass_at_k_value is not None:
                        metrics[f"pass{pass_at_k_value}"] += (
                            chunk_rewards[:pass_at_k_value].max().float().item()
                        )
                elif n_samples_per_prompt > 1:
                    metrics[f"pass{n_samples_per_prompt}"] += (
                        chunk_rewards.max().float().item()
                    )
                if not eval_name or log_pass_at_1:
                    metrics["pass1"] += chunk_rewards.mean().float().item()
                metrics["count"] += 1
            logs = {}
            if eval_name:
                if log_pass_at_1:
                    logs[f"{eval_name}/pass@1"] = (
                        global_metrics["pass1"] / global_metrics["count"]
                    )
                if log_pass_at_k and pass_at_k_value is not None:
                    logs[f"{eval_name}/pass@{pass_at_k_value}"] = (
                        global_metrics[f"pass{pass_at_k_value}"]
                        / global_metrics["count"]
                    )
            else:
                for (datasource, metrics) in global_metrics.items():
                    logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                        metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                    )
                    logs[f"eval_{datasource}_pass1"] = (
                        metrics["pass1"] / metrics["count"]
                    )
            eval_logs = {
                f"eval/{k}": v for (k, v) in logs.items() if k != "global_step"
            }
            wandb_eval_logs = {
                f"eval/{k}": v
                for (k, v) in self._wandb_eval_metric_aliases(logs).items()
                if k != "global_step"
            }
            if self._wandb is not None:
                if log_to_wandb:
                    self._wandb_log_at_global_step(wandb_eval_logs, global_step)
            elif self._tensorboard is not None and log_to_tensorboard:
                for (k, v) in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        console_logs = {**eval_logs, "global_step": global_step}
        logger.info(
            f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {console_logs}"
        )
        return {"logs": logs, "eval_logs": eval_logs, "wandb_logs": wandb_eval_logs}

    def prepare_datasets(self):
        from reproduction.data import load_source

        args = self.args
        strategy = self.strategy
        train_data = load_source(
            args.prompt_data,
            split=self.prompt_split,
            config=args.prompt_data_config,
            revision=args.prompt_data_revision,
            max_samples=args.max_samples,
        )
        train_data = train_data.shuffle(seed=args.seed)
        prompts_dataset = PromptDataset(
            train_data,
            self.tokenizer,
            strategy,
            input_template=args.input_template,
            prompt_type=args.prompt_type,
        )
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.vllm_generate_batch_size, True, True
        )
        eval_specs = []
        if getattr(args, "eval_dataset", None):
            eval_specs.append(
                {
                    "dataset": args.eval_dataset,
                    "name": getattr(args, "eval_dataset_name", None)
                    or _infer_eval_dataset_name(args.eval_dataset, "eval"),
                    "split": self.eval_split,
                    "input_key": args.eval_input_key,
                    "label_key": args.eval_label_key,
                }
            )
        if getattr(args, "eval_dataset2", None):
            eval_specs.append(
                {
                    "dataset": args.eval_dataset2,
                    "name": getattr(args, "eval_dataset2_name", None)
                    or _infer_eval_dataset_name(args.eval_dataset2, "eval2"),
                    "split": getattr(args, "eval_split2", None) or self.eval_split,
                    "input_key": getattr(args, "eval_input_key2", args.eval_input_key),
                    "label_key": getattr(args, "eval_label_key2", args.eval_label_key),
                }
            )
        eval_dataloaders = []
        for eval_spec in eval_specs:
            eval_data = load_source(eval_spec["dataset"], split=eval_spec["split"])
            eval_dataset = PromptDataset(
                eval_data,
                self.tokenizer,
                strategy,
                input_template=args.input_template,
                prompt_type=args.prompt_type,
                input_key=eval_spec["input_key"],
                label_key=eval_spec["label_key"],
            )
            eval_dataloaders.append(
                {
                    "name": eval_spec["name"],
                    "dataloader": strategy.setup_dataloader(
                        eval_dataset, 1, True, False
                    ),
                }
            )
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloaders = eval_dataloaders
        self.eval_dataloader = (
            eval_dataloaders[0]["dataloader"] if eval_dataloaders else None
        )
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )

    def get_max_steps(self):
        return self.max_steps


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

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
        self.samples_generator = self.generator_cls(
            self.vllm_engines, self.strategy, self.tokenizer, self.prompt_max_len
        )
        self.experience_maker = RemoteExperienceMaker(
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
                rollout_bundle = self.samples_generator.generate_samples(
                    rand_prompts, labels, global_step=steps, **self.generate_kwargs
                )
                pbar.update()
                rollout_samples = rollout_bundle
                status = {}
                experiences = []
                sample0 = [""]
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
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                actor_status = self.ppo_train(steps)
                status.update(actor_status)
                status.update(reward_status)
                if "kl" in actor_status:
                    self.kl_ctl.update(
                        actor_status["kl"],
                        args.rollout_batch_size * args.n_samples_per_prompt,
                    )
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                status = self._filter_training_status(status)
                logger.info(f"✨ Global step {steps}: {status}")
                if len(experiences) > 0:
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
