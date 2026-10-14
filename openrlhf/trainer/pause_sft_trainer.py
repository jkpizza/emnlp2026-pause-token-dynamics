import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.utils.distributed_sampler import DistributedSampler


def _single_token_id(tokenizer, token):
    if tokenizer is None:
        return None
    converted = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if converted is None or converted == unk_id:
        return None
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def compute_pause_sft_loss(
    per_token_logps,
    labels,
    loss_mask,
    pause_token_id=None,
    predict_pause_tokens=False,
):
    """Mean CE over active text targets and, optionally, pause targets."""
    ce = -per_token_logps
    active = loss_mask.bool()
    pause_mask = torch.zeros_like(active, dtype=torch.bool)
    if pause_token_id is not None:
        pause_mask = active & labels.eq(int(pause_token_id))
    text_mask = active & ~pause_mask
    weight_mask = (active if predict_pause_tokens else text_mask).to(ce.dtype)
    effective_loss_weight_sum = weight_mask.sum()
    if float(effective_loss_weight_sum.detach().item()) <= 0.0:
        raise RuntimeError("No active text or enabled pause targets in this SFT batch.")
    loss = (ce * weight_mask).sum() / effective_loss_weight_sum

    stats = {"effective_loss_weight_sum": float(effective_loss_weight_sum.detach().item())}
    text_count = int(text_mask.sum().detach().item())
    stats["text_token_count"] = float(text_count)
    stats["text_loss_component"] = float(ce[text_mask].mean().detach().item()) if text_count else 0.0
    pause_count = int(pause_mask.sum().detach().item())
    stats["pause_token_active_count"] = float(pause_count)
    stats["pause_token_loss_weight"] = float(predict_pause_tokens)
    stats["pause_token_loss_component"] = float(ce[pause_mask].mean().detach().item()) if pause_count > 0 else 0.0
    return loss, stats


def validate_pause_token_loss_mask(labels, loss_mask, pause_token_id, allow_pause_token_loss=False):
    if pause_token_id is None:
        return loss_mask, {"pause_token_count": 0.0, "pause_token_loss_masked_count": 0.0}

    pause_positions = labels.eq(int(pause_token_id))
    active_pause_positions = pause_positions & loss_mask.bool()
    active_pause_count = int(active_pause_positions.sum().detach().item())
    if active_pause_count > 0 and not allow_pause_token_loss:
        raise RuntimeError(
            "Pause token loss mask must be applied by PauseSFTDataset before training/eval. "
            f"Found {active_pause_count} active pause-token targets in the shifted loss mask."
        )
    masked_pause_positions = pause_positions & ~loss_mask.bool()
    return loss_mask, {
        "pause_token_count": float(pause_positions.sum().detach().item()),
        "pause_token_loss_masked_count": float(masked_pause_positions.sum().detach().item()),
    }


class PauseSFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.pause_token = getattr(self.args, "pause_token", "<|PAUSE|>")
        self.pause_token_id = _single_token_id(tokenizer, self.pause_token)
        if self.pause_token_id is None:
            raise ValueError(f"Pause token {self.pause_token!r} must exist and tokenize to exactly one id.")
        self.predict_pause_tokens = bool(getattr(self.args, "predict_pause_tokens", False))

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        # save once per epoch
        elif args.save_steps == 0:
            args.save_steps = num_update_steps_per_epoch

        print_examples = True
        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for inputs, attention_masks, loss_masks in self.train_dataloader:
                if print_examples:
                    if self.strategy.is_rank_0():
                        input_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                        self.strategy.print(input_text)
                        print_examples = False
                        self.strategy.print("--------------------------------")
                        # import pdbpp as pdb; pdb.set_trace()
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                per_token_log_probs, output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )
                # import pdbpp as pdb; pdb.set_trace()

                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                labels = inputs[:, 1:]
                shifted_loss_mask = loss_mask[:, 1:]
                shifted_loss_mask, pause_mask_stats = validate_pause_token_loss_mask(
                    labels,
                    shifted_loss_mask,
                    self.pause_token_id,
                    allow_pause_token_loss=self.predict_pause_tokens,
                )
                gpt_loss, loss_stats = compute_pause_sft_loss(
                    per_token_log_probs,
                    labels,
                    shifted_loss_mask,
                    pause_token_id=self.pause_token_id,
                    predict_pause_tokens=self.predict_pause_tokens,
                )
                loss_stats.update(pause_mask_stats)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                logs_dict.update(loss_stats)
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(
                    {
                        "loss": logs_dict["gpt_loss"],
                        "lr": logs_dict["lr"],
                    }
                )
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                    if getattr(args, "max_steps", -1) > 0 and global_step >= args.max_steps:
                        break

                step += 1

            epoch_bar.update()
            if getattr(args, "max_steps", -1) > 0 and step // self.strategy.accumulated_gradient >= args.max_steps:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when eval_dataloader is not None and len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            loss_stats_sum = {}
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for inputs, attention_masks, loss_masks in eval_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                per_token_log_probs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                labels = inputs[:, 1:]
                shifted_loss_mask = loss_mask[:, 1:]
                shifted_loss_mask, pause_mask_stats = validate_pause_token_loss_mask(
                    labels,
                    shifted_loss_mask,
                    self.pause_token_id,
                    allow_pause_token_loss=self.predict_pause_tokens,
                )
                loss, loss_stats = compute_pause_sft_loss(
                    per_token_log_probs,
                    labels,
                    shifted_loss_mask,
                    pause_token_id=self.pause_token_id,
                    predict_pause_tokens=self.predict_pause_tokens,
                )
                loss_stats.update(pause_mask_stats)

                times += 1
                loss_sum += loss.item()
                for key, value in loss_stats.items():
                    loss_stats_sum[key] = loss_stats_sum.get(key, 0.0) + float(value)
                bar_dict = {"eval gpt_loss": loss_sum / times}
                for key, value in loss_stats_sum.items():
                    bar_dict[f"eval {key}"] = value / times
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix({"loss": logs["eval gpt_loss"]})

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
