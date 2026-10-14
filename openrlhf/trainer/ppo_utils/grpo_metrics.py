"""Training reward summaries and pause update diagnostics."""
from typing import Any, Dict, List, Sequence
import torch

PAUSE_TRAINING_METRICS = frozenset(
    {
        "pause/penalty_scale",
        "pause/clip_penalty",
        "pause/over_target_length",
        "pause/over_target_length_penalty",
        "pause/total_length_penalty",
        "pause/penalty_correct_gate",
        "pause/penalty_active",
        "pause/effective_clip_penalty_coef",
        "pause/effective_over_target_token_penalty_coef",
    }
)


def should_skip_pause_training_metric(key: Any) -> bool:
    key = str(key)
    return (
        key.startswith("pause/")
        and key not in PAUSE_TRAINING_METRICS
        and not key.startswith("pause/posthoc_")
    )


def reward_group_metrics(
    rewards: Sequence[float], n_samples_per_prompt: int, threshold: float = 0.5
) -> Dict[str, float]:
    group_size = max(1, int(n_samples_per_prompt))
    usable = len(rewards) // group_size * group_size
    rewards = [float(x) for x in rewards[:usable]]
    if usable == 0:
        return {
            "reward/mean": 0.0,
            "reward/pass1": 0.0,
            "reward/mixed_ratio": 0.0,
            "reward/all_correct_ratio": 0.0,
            "reward/all_wrong_ratio": 0.0,
            "reward/group_count": 0.0,
        }
    binary = [1 if value > float(threshold) else 0 for value in rewards]
    group_counts = [
        sum(binary[start : start + group_size])
        for start in range(0, usable, group_size)
    ]
    group_count = len(group_counts)
    out = {
        "reward/mean": float(sum(rewards) / len(rewards)),
        "reward/pass1": float(sum(rewards) / len(rewards)),
        "reward/mixed_ratio": float(
            sum((1 for c in group_counts if 0 < c < group_size)) / group_count
        ),
        "reward/all_correct_ratio": float(
            sum((1 for c in group_counts if c == group_size)) / group_count
        ),
        "reward/all_wrong_ratio": float(
            sum((1 for c in group_counts if c == 0)) / group_count
        ),
        "reward/group_count": float(group_count),
    }
    for c in range(group_size + 1):
        out[f"reward/c_hist_{c}"] = float(
            sum((1 for value in group_counts if value == c))
        )
    return out


def rewards_from_samples(samples: Sequence[Any]) -> List[float]:
    rewards: List[float] = []
    for sample in samples:
        value = getattr(sample, "scores", None)
        if value is None:
            value = getattr(sample, "rewards", None)
        if value is None and getattr(sample, "info", None):
            value = sample.info.get("score")
            if value is None:
                value = sample.info.get("reward")
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            rewards.extend(
                (float(x) for x in value.detach().float().cpu().reshape(-1).tolist())
            )
        else:
            rewards.append(float(value))
    return rewards


def masked_mean_safe(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a masked scalar mean, including empty diagnostic windows."""
    mask = mask.to(device=tensor.device, dtype=tensor.dtype)
    denom = mask.sum()
    if denom.item() == 0:
        return tensor.new_tensor(0.0)
    return (tensor * mask).sum() / denom
