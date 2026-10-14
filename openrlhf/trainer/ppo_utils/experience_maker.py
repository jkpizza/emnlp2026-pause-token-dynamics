from openrlhf.trainer.ppo_utils.pause_schedule import scheduled_pause_ratio
import random
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Any, List, Optional, Tuple, Union, Dict
import ray
import torch
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import (
    get_minimum_num_micro_batch_size,
    get_seqlen_balanced_partitions,
)
from openrlhf.utils.split_steps import split_atomic_reasoning_step_spans
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None
    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None
    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None
    scores: torch.Tensor = None
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for (field, value) in self.__dict__.items():
            if isinstance(value, dict):
                setattr(
                    self, field, {key: to(val, device) for (key, val) in value.items()}
                )
            else:
                setattr(self, field, to(value, device))
        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for (field, value) in self.__dict__.items():
            if isinstance(value, dict):
                setattr(
                    self, field, {key: pin_memory(val) for (key, val) in value.items()}
                )
            else:
                setattr(self, field, pin_memory(value))
        return self

    @staticmethod
    def select(
        experiences: List["Experience"], fields: List[str]
    ) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(
        items: List, pad_value: int = 0
    ) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            for d in items:
                for (key, value) in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            return {
                key: Experience._merge_item(values, pad_value)
                for (key, values) in result.items()
            }
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(
        experiences_list: List["Experience"], pad_token_id
    ) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()
        field_names = [f.name for f in fields(Experience)]
        result = {}
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)
        return Experience(**result)


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(
        self, all_prompts: List[str], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        rollout_samples = self._generate_vllm(
            all_prompts, all_labels, **generate_kwargs
        )
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        return rollout_samples

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            return self.tokenizer(
                texts, add_special_tokens=False, max_length=max_length, truncation=True
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for (k, v) in batch.items()}

    def _generate_vllm(
        self, all_prompts: List[str], all_labels, **kwargs
    ) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05),
            include_stop_str_in_output=True,
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length
        n_samples_per_prompt = kwargs.pop(
            "n_samples_per_prompt", args.n_samples_per_prompt
        )
        all_prompts = sum(
            [[prompt] * n_samples_per_prompt for prompt in all_prompts], []
        )
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False
        )["input_ids"]
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for (i, llm) in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[
                i * batch_size : (i + 1) * batch_size
            ]
            refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params, prompt_token_ids=prompt_token_ids
                )
            )
        ray.get(refs)
        all_output_refs = []
        for (i, llm) in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])
        samples_list = []
        final_outputs = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]
            input_ids = list(output.prompt_token_ids) + list(
                output.outputs[0].token_ids
            )
            attention_mask = [1] * len(input_ids)
            sequences = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            final_outputs.append(output.outputs[0].text)
            action_mask = torch.zeros_like(attention_mask)
            response_length = len(output.outputs[0].token_ids)
            action_mask[
                len(output.prompt_token_ids) : len(output.prompt_token_ids)
                + response_length
            ] = 1
            rollout_log_probs = None
            if self.strategy.args.enable_vllm_is_correction:
                rollout_log_probs = []
                response_ids = list(output.outputs[0].token_ids)
                for (i, logprob) in enumerate(output.outputs[0].logprobs):
                    rollout_log_probs.append(logprob[response_ids[i]].logprob)
                rollout_log_probs = torch.tensor(
                    [0.0] * len(list(output.prompt_token_ids)) + rollout_log_probs
                )
                rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")
            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            total_length = attention_mask.float().sum()
            is_clipped = response_length >= max_response_length
            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
            }
            rollout_samples = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                rollout_log_probs=rollout_log_probs.unsqueeze(0)
                if rollout_log_probs is not None
                else None,
                prompts=[prompt],
                labels=[label],
                info=info,
            )
            samples_list.append(rollout_samples)
        logger.info(
            f"🚀 Rollout sample: {all_prompts[0] + all_outputs[0].outputs[0].text}"
        )
        from evaluation.scripts.utils import extract_all_answers
        from evaluation.scripts.eval_utils import math_equal

        def is_answer_equal(p_ans, ans):
            if p_ans is None or ans is None:
                return False
            return str(p_ans).strip() == str(ans).strip()

        for (i, sample) in enumerate(samples_list):
            full_text = final_outputs[i]
            label = all_labels[i]
            reward_val = 0.0
            if label is not None:
                predicted_answers = extract_all_answers(full_text, args.answer_trigger)
                if predicted_answers is not None:
                    for p_ans in predicted_answers:
                        if p_ans is not None and (
                            is_answer_equal(p_ans, label) or math_equal(p_ans, label)
                        ):
                            reward_val = 1.0
                            break
            reward_tensor = torch.tensor(
                [reward_val], dtype=torch.float32, device="cpu"
            )
            sample.rewards = reward_tensor.clone()
            sample.scores = reward_tensor.clone()
            sample.info["reward"] = reward_tensor.clone()
            sample.info["score"] = reward_tensor.clone()
        return samples_list


class PauseGRPOSamplesGenerator(SamplesGenerator):
    """Single-policy GRPO rollout generator that inserts pause tokens after rollout."""

    PAUSE_SCALAR_METRIC_DEFAULTS = {
        "pause/tokenization_text_reencode_match": 0.0,
        "pause/tokenization_vllm_match": 0.0,
        "pause/tokenization_text_token_count": 0.0,
        "pause/tokenization_vllm_token_count": 0.0,
        "pause/tokenization_used_vllm_ids_as_source": 1.0,
        "pause/tokenization_used_prefix_fast_path": 0.0,
        "pause/tokenization_vllm_trailing_text_invisible_ids": 0.0,
        "pause/tokenization_used_vllm_trailing_ids": 0.0,
        "pause/tokenization_offsets_from_vllm_prefix": 0.0,
        "pause/tokenization_offset_coverage_ratio": 1.0,
        "pause/tokenization_prefix_decode_skip_special": 0.0,
        "pause/tokenization_uncovered_chars": 0.0,
        "pause/tokenization_used_text_reencode_fallback": 0.0,
        "pause/span_count": 0.0,
        "pause/raw_insertion_boundary_count": 0.0,
        "pause/collapsed_insertion_boundary_count": 0.0,
        "pause/boundary_inside_token_count": 0.0,
    }
    PAUSE_SCALAR_METRIC_DEFAULTS.update(
        {
            "pause/tokenization_used_fast_offsets": 0.0,
            "pause/tokenization_used_prefix_fast_path": 0.0,
            "pause/tokenization_chunked_prefix_decode_used": 0.0,
            "pause/tokenization_offsets_from_vllm_prefix": 0.0,
            "pause/tokenization_prefix_decode_calls": 0.0,
            "pause/tokenization_prefix_decode_block_size": 0.0,
            "pause/tokenization_prefix_decode_skip_special": 0.0,
            "pause/tokenization_offset_coverage_ratio": 1.0,
            "pause/tokenization_uncovered_chars": 0.0,
            "pause/tokenization_vllm_trailing_text_invisible_ids": 0.0,
            "pause/tokenization_used_id_alignment": 0.0,
            "pause/tokenization_id_alignment_boundary_count": 0.0,
            "pause/tokenization_id_alignment_mapped_boundary_count": 0.0,
            "pause/tokenization_id_alignment_unmapped_boundary_count": 0.0,
            "pause/tokenization_id_alignment_fallback_boundary_count": 0.0,
            "pause/tokenization_id_alignment_fallback_success_count": 0.0,
            "pause/unusable_insertion_boundary_count": 0.0,
            "pause/posthoc_sample_count": 0.0,
            "pause/posthoc_rollout_ratio": 1.0,
            "pause/posthoc_pause_applied": 0.0,
            "pause/posthoc_pause_applied_sample_count": 0.0,
            "pause/posthoc_pause_skipped_sample_count": 0.0,
        }
    )

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _decode_token_ids(self, token_ids: List[int], skip_special_tokens: bool) -> str:
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )

    def _encode_response_once_with_offsets(
        self, response_text: str
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        if not response_text:
            return ([], [])
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "Pause GRPO step-token alignment requires a fast tokenizer. Remove --disable_fast_tokenizer."
            )
        encoded = self.tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            offsets = offsets[0]
        return ([int(x) for x in input_ids], [(int(s), int(e)) for (s, e) in offsets])

    @staticmethod
    def _char_start_to_token_index(
        offsets: List[Tuple[int, int]], char_start: int
    ) -> Optional[int]:
        if not offsets:
            return None
        if char_start <= 0:
            for (idx, (token_start, token_end)) in enumerate(offsets):
                if token_end > token_start:
                    return idx
            return 0
        for (idx, (token_start, token_end)) in enumerate(offsets):
            if token_end <= token_start:
                continue
            if token_end > char_start:
                return idx
        return len(offsets)

    @staticmethod
    def _build_text_to_vllm_boundary_map(
        text_ids: List[int], vllm_ids: List[int]
    ) -> Dict[int, int]:
        """
        Return text-token-boundary index -> vLLM-token-boundary index.

        Boundaries are positions between tokens: 0 is before the first token,
        len(ids) is after the last token. Only exact token-id equal regions are
        mapped. Suffix mappings intentionally run after prefix mappings so a
        shared boundary around inserted vLLM ids maps to the boundary before the
        next visible text token.
        """
        text_ids = [int(x) for x in text_ids]
        vllm_ids = [int(x) for x in vllm_ids]
        text_len = len(text_ids)
        vllm_len = len(vllm_ids)
        min_len = min(text_len, vllm_len)
        boundary_map: Dict[int, int] = {0: 0}
        common_prefix = 0
        while (
            common_prefix < min_len
            and text_ids[common_prefix] == vllm_ids[common_prefix]
        ):
            common_prefix += 1
        common_suffix = 0
        while (
            common_prefix + common_suffix < min_len
            and text_ids[text_len - common_suffix - 1]
            == vllm_ids[vllm_len - common_suffix - 1]
        ):
            common_suffix += 1

        def add_equal_boundary_block(
            text_start: int, vllm_start: int, length: int
        ) -> None:
            for offset in range(length + 1):
                boundary_map[text_start + offset] = vllm_start + offset

        add_equal_boundary_block(0, 0, common_prefix)
        text_mid_start = common_prefix
        text_mid_end = text_len - common_suffix
        vllm_mid_start = common_prefix
        vllm_mid_end = vllm_len - common_suffix
        if text_mid_start < text_mid_end and vllm_mid_start < vllm_mid_end:
            matcher = SequenceMatcher(
                None,
                text_ids[text_mid_start:text_mid_end],
                vllm_ids[vllm_mid_start:vllm_mid_end],
                autojunk=False,
            )
            for (tag, text_i1, text_i2, vllm_j1, vllm_j2) in matcher.get_opcodes():
                if tag != "equal":
                    continue
                length = min(text_i2 - text_i1, vllm_j2 - vllm_j1)
                add_equal_boundary_block(
                    text_mid_start + text_i1, vllm_mid_start + vllm_j1, length
                )
        if common_suffix:
            add_equal_boundary_block(
                text_len - common_suffix, vllm_len - common_suffix, common_suffix
            )
        return boundary_map

    def _map_char_starts_by_chunked_prefix_decode(
        self,
        response_text: str,
        response_token_ids: List[int],
        char_starts: List[int],
        collect_metrics: bool = True,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Map step-start character offsets to vLLM token indices.

        This avoids the old O(n^2) path:
            decode(ids[:1]), decode(ids[:2]), ..., decode(ids[:n])

        Instead:
            1. choose skip_special_tokens once using full decode
            2. build sparse prefix checkpoints every B tokens
            3. binary-search inside the relevant chunk for each step boundary

        Complexity is roughly:
            O(n / B + num_boundaries * log B) decode calls
        instead of:
            O(2n) decode calls with repeatedly growing prefixes.
        """
        vllm_ids = [int(x) for x in response_token_ids]
        n = len(vllm_ids)
        text_len = len(response_text)
        if n == 0:
            metrics = {}
            if collect_metrics:
                metrics = {
                    "pause/tokenization_chunked_prefix_decode_used": 1.0,
                    "pause/tokenization_prefix_decode_calls": 0.0,
                    "pause/tokenization_offset_coverage_ratio": 0.0,
                }
            return ([], metrics)
        if text_len == 0:
            metrics = {}
            if collect_metrics:
                metrics = {
                    "pause/tokenization_chunked_prefix_decode_used": 1.0,
                    "pause/tokenization_prefix_decode_calls": 0.0,
                    "pause/tokenization_offset_coverage_ratio": 0.0,
                }
            return ([0 for _ in char_starts], metrics)
        block_size = int(
            getattr(self.strategy.args, "pause_prefix_decode_block_size", 128)
        )
        block_size = max(16, block_size)
        cache: Dict[Tuple[int, bool], int] = {}
        decode_calls = 0

        def covered_chars(end_idx: int, skip_special_tokens: bool) -> int:
            nonlocal decode_calls
            end_idx = max(0, min(int(end_idx), n))
            key = (end_idx, bool(skip_special_tokens))
            if key in cache:
                return cache[key]
            decode_calls += 1
            decoded_prefix = self._decode_token_ids(
                vllm_ids[:end_idx], skip_special_tokens=skip_special_tokens
            )
            covered = self._common_prefix_len(response_text, decoded_prefix)
            covered = max(0, min(covered, text_len))
            cache[key] = covered
            return covered

        full_false = covered_chars(n, False)
        full_true = covered_chars(n, True)
        skip_special = bool(full_true > full_false)
        full_covered = max(full_false, full_true)

        def cov(end_idx: int) -> int:
            return covered_chars(end_idx, skip_special)

        checkpoints: List[Tuple[int, int]] = [(0, 0)]
        last_cov = 0
        end = block_size
        while end < n:
            c = cov(end)
            c = max(last_cov, min(c, text_len))
            checkpoints.append((end, c))
            last_cov = c
            end += block_size
        final_cov = max(last_cov, cov(n))
        final_cov = min(final_cov, text_len)
        checkpoints.append((n, final_cov))
        positions: List[int] = []
        for char_start in char_starts:
            char_start = int(char_start)
            if char_start <= 0:
                positions.append(0)
                continue
            if char_start >= text_len:
                positions.append(n)
                continue
            low_end = 0
            low_cov = 0
            high_end = n
            found_chunk = False
            for (ckpt_end, ckpt_cov) in checkpoints[1:]:
                if ckpt_cov > char_start:
                    high_end = ckpt_end
                    found_chunk = True
                    break
                low_end = ckpt_end
                low_cov = ckpt_cov
            if not found_chunk:
                positions.append(n)
                continue
            lo = low_end + 1
            hi = high_end
            while lo < hi:
                mid = (lo + hi) // 2
                mid_cov = max(low_cov, cov(mid))
                if mid_cov > char_start:
                    hi = mid
                else:
                    lo = mid + 1
            positions.append(max(0, min(n, lo - 1)))
        metrics = {}
        if collect_metrics:
            metrics = {
                "pause/tokenization_chunked_prefix_decode_used": 1.0,
                "pause/tokenization_offsets_from_vllm_prefix": 1.0,
                "pause/tokenization_prefix_decode_calls": float(decode_calls),
                "pause/tokenization_prefix_decode_block_size": float(block_size),
                "pause/tokenization_prefix_decode_skip_special": float(skip_special),
                "pause/tokenization_offset_coverage_ratio": float(
                    full_covered / max(1, text_len)
                ),
                "pause/tokenization_uncovered_chars": float(
                    max(0, text_len - full_covered)
                ),
            }
        return (positions, metrics)

    @classmethod
    def _complete_pause_metrics(
        cls, metrics: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Return a fixed-schema scalar metric dict.

        Important:
        Experience.concat_experiences() does not fill missing info keys.
        Therefore every per-sample Experience must contain exactly the same
        pause diagnostic keys.
        """
        out = dict(cls.PAUSE_SCALAR_METRIC_DEFAULTS)
        metrics = metrics or {}
        for key in out.keys():
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    value = 0.0
                else:
                    value = value.detach().float().mean().cpu().item()
            elif isinstance(value, bool):
                value = float(value)
            else:
                value = float(value)
            out[key] = value
        unknown_keys = sorted(set(metrics.keys()) - set(out.keys()))
        if unknown_keys:
            logger.warning(
                "Ignoring unknown branch-specific pause metric keys to keep Experience.info schema stable: %s",
                unknown_keys,
            )
        return out

    @staticmethod
    def _assert_same_info_keys(samples: List[Experience]) -> None:
        if not samples:
            return
        expected = set(samples[0].info.keys())
        for (idx, sample) in enumerate(samples[1:], start=1):
            keys = set(sample.info.keys())
            if keys != expected:
                missing = sorted(expected - keys)
                extra = sorted(keys - expected)
                raise RuntimeError(
                    f"Pause sample info schema mismatch before concat: sample_idx={idx}, missing={missing}, extra={extra}"
                )

    @staticmethod
    def _scheduled_posthoc_rollout_ratio(
        start_ratio: float,
        end_ratio: float,
        warmup_start_step: int,
        warmup_end_step: int,
        global_step: int,
    ) -> float:
        return scheduled_pause_ratio(
            start_probability=start_ratio,
            end_probability=end_ratio,
            warmup_start_step=warmup_start_step,
            warmup_end_step=warmup_end_step,
            global_step=global_step,
        )

    def _current_posthoc_rollout_ratio(self, global_step: int) -> float:
        args = self.strategy.args
        ratio = self._scheduled_posthoc_rollout_ratio(
            start_ratio=getattr(args, "pause_posthoc_rollout_start_ratio", 1.0),
            end_ratio=getattr(args, "pause_posthoc_rollout_end_ratio", 1.0),
            warmup_start_step=getattr(
                args, "pause_posthoc_rollout_ratio_warmup_start_step", 0
            ),
            warmup_end_step=getattr(
                args, "pause_posthoc_rollout_ratio_warmup_end_step", 0
            ),
            global_step=global_step,
        )
        return max(0.0, min(1.0, float(ratio)))

    @staticmethod
    def _sample_posthoc_pause_routes(
        num_samples: int, ratio: float, seed: int, global_step: int
    ) -> List[bool]:
        if num_samples <= 0:
            return []
        ratio = max(0.0, min(1.0, float(ratio)))
        if ratio <= 0.0:
            return [False] * num_samples
        if ratio >= 1.0:
            return [True] * num_samples
        selected_count = max(0, min(num_samples, int(num_samples * ratio + 0.5)))
        indices = list(range(num_samples))
        rng = random.Random(int(seed) + int(global_step) + 1000003)
        rng.shuffle(indices)
        selected = set(indices[:selected_count])
        return [idx in selected for idx in range(num_samples)]

    def _posthoc_pause_routes_for_batch(
        self, num_samples: int, global_step: int
    ) -> Tuple[float, List[bool]]:
        args = self.strategy.args
        if not getattr(args, "pause_posthoc_rollout_ratio_schedule", False):
            return (1.0, [True] * max(0, int(num_samples)))
        ratio = self._current_posthoc_rollout_ratio(global_step)
        routes = self._sample_posthoc_pause_routes(
            num_samples=num_samples,
            ratio=ratio,
            seed=getattr(args, "seed", 0),
            global_step=global_step,
        )
        return (ratio, routes)

    def _stamp_posthoc_pause_ratio_metrics(
        self,
        sample: Experience,
        *,
        pause_applied: bool,
        posthoc_ratio: float,
        applied_sample_count: int,
        skipped_sample_count: int,
    ) -> None:
        sample.info["pause/posthoc_rollout_ratio"] = torch.tensor(
            [float(posthoc_ratio)], dtype=torch.float32, device="cpu"
        )
        sample.info["pause/posthoc_pause_applied"] = torch.tensor(
            [1.0 if pause_applied else 0.0], dtype=torch.float32, device="cpu"
        )
        sample.info["pause/posthoc_pause_applied_sample_count"] = torch.tensor(
            [float(applied_sample_count)], dtype=torch.float32, device="cpu"
        )
        sample.info["pause/posthoc_pause_skipped_sample_count"] = torch.tensor(
            [float(skipped_sample_count)], dtype=torch.float32, device="cpu"
        )

    @staticmethod
    def _filter_complete_pause_groups(
        sample_slots: List[Optional[Experience]],
        original_outputs: List[str],
        n_samples_per_prompt: int,
    ) -> Tuple[List[Experience], List[str], int]:
        samples_list: List[Experience] = []
        valid_original_outputs: List[str] = []
        dropped_groups = 0
        for group_start in range(0, len(sample_slots), n_samples_per_prompt):
            group_samples = sample_slots[
                group_start : group_start + n_samples_per_prompt
            ]
            group_outputs = original_outputs[
                group_start : group_start + n_samples_per_prompt
            ]
            if len(group_samples) < n_samples_per_prompt or any(
                (sample is None for sample in group_samples)
            ):
                dropped_groups += 1
                continue
            samples_list.extend(group_samples)
            valid_original_outputs.extend(group_outputs)
        return (samples_list, valid_original_outputs, dropped_groups)

    def _pause_token_ids(self) -> List[int]:
        args = self.strategy.args
        pause_ids = self.tokenizer.encode(args.pause_token, add_special_tokens=False)
        if len(pause_ids) != 1:
            raise ValueError(
                f"{args.pause_token!r} must tokenize to exactly one id, got {pause_ids}"
            )
        tokens_per_step = max(1, int(getattr(args, "pause_tokens_per_step", 1)))
        return pause_ids * tokens_per_step

    def _map_step_starts_to_token_indices(
        self,
        response_text: str,
        response_token_ids: List[int],
        span_starts: List[int],
        collect_metrics: bool = True,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Source of truth is always vLLM completion.token_ids.

        response_text is used only to locate step boundaries.
        """
        vllm_ids = [int(x) for x in response_token_ids]
        (text_ids, text_offsets) = self._encode_response_once_with_offsets(
            response_text
        )
        metrics = {}
        if collect_metrics:
            metrics = {
                "pause/tokenization_text_reencode_match": float(text_ids == vllm_ids),
                "pause/tokenization_text_token_count": float(len(text_ids)),
                "pause/tokenization_vllm_token_count": float(len(vllm_ids)),
                "pause/tokenization_used_vllm_ids_as_source": 1.0,
                "pause/tokenization_used_fast_offsets": 0.0,
                "pause/tokenization_used_prefix_fast_path": 0.0,
                "pause/tokenization_chunked_prefix_decode_used": 0.0,
                "pause/tokenization_offsets_from_vllm_prefix": 0.0,
                "pause/tokenization_prefix_decode_calls": 0.0,
                "pause/tokenization_prefix_decode_block_size": 0.0,
                "pause/tokenization_prefix_decode_skip_special": 0.0,
                "pause/tokenization_offset_coverage_ratio": 1.0,
                "pause/tokenization_uncovered_chars": 0.0,
                "pause/tokenization_vllm_trailing_text_invisible_ids": 0.0,
                "pause/tokenization_used_id_alignment": 0.0,
                "pause/tokenization_id_alignment_boundary_count": 0.0,
                "pause/tokenization_id_alignment_mapped_boundary_count": 0.0,
                "pause/tokenization_id_alignment_unmapped_boundary_count": 0.0,
                "pause/tokenization_id_alignment_fallback_boundary_count": 0.0,
                "pause/tokenization_id_alignment_fallback_success_count": 0.0,
            }
        if not vllm_ids:
            return ([], metrics)
        text_boundaries = [
            self._char_start_to_token_index(text_offsets, start)
            for start in span_starts
        ]
        if text_ids == vllm_ids:
            if collect_metrics:
                metrics["pause/tokenization_used_fast_offsets"] = 1.0
            return ([p for p in text_boundaries if p is not None], metrics)
        if (
            text_ids
            and len(vllm_ids) >= len(text_ids)
            and (vllm_ids[: len(text_ids)] == text_ids)
        ):
            extra = len(vllm_ids) - len(text_ids)
            offsets = text_offsets + [(len(response_text), len(response_text))] * extra
            positions = [
                self._char_start_to_token_index(offsets, start) for start in span_starts
            ]
            if collect_metrics:
                metrics["pause/tokenization_used_prefix_fast_path"] = 1.0
                metrics["pause/tokenization_vllm_trailing_text_invisible_ids"] = float(
                    extra
                )
            return ([p for p in positions if p is not None], metrics)
        boundary_map = self._build_text_to_vllm_boundary_map(text_ids, vllm_ids)
        positions: List[Optional[int]] = []
        fallback_slots: List[int] = []
        fallback_char_starts: List[int] = []
        mapped_count = 0
        for (slot_idx, (text_boundary, char_start)) in enumerate(
            zip(text_boundaries, span_starts)
        ):
            if text_boundary is None:
                positions.append(None)
                continue
            vllm_boundary = boundary_map.get(int(text_boundary))
            if vllm_boundary is None:
                positions.append(None)
                fallback_slots.append(slot_idx)
                fallback_char_starts.append(int(char_start))
                continue
            mapped_count += 1
            positions.append(int(vllm_boundary))
        fallback_success_count = 0
        if fallback_char_starts:
            (
                fallback_positions,
                slow_metrics,
            ) = self._map_char_starts_by_chunked_prefix_decode(
                response_text=response_text,
                response_token_ids=vllm_ids,
                char_starts=fallback_char_starts,
                collect_metrics=collect_metrics,
            )
            if collect_metrics:
                metrics.update(slow_metrics)
            for (slot_idx, fallback_pos) in zip(fallback_slots, fallback_positions):
                fallback_pos = int(fallback_pos)
                positions[slot_idx] = fallback_pos
                if 0 <= fallback_pos < len(vllm_ids):
                    fallback_success_count += 1
        if collect_metrics:
            boundary_count = sum(
                (1 for boundary in text_boundaries if boundary is not None)
            )
            fallback_count = len(fallback_char_starts)
            metrics["pause/tokenization_used_id_alignment"] = 1.0
            metrics["pause/tokenization_id_alignment_boundary_count"] = float(
                boundary_count
            )
            metrics["pause/tokenization_id_alignment_mapped_boundary_count"] = float(
                mapped_count
            )
            metrics["pause/tokenization_id_alignment_unmapped_boundary_count"] = float(
                fallback_count
            )
            metrics["pause/tokenization_id_alignment_fallback_boundary_count"] = float(
                fallback_count
            )
            metrics["pause/tokenization_id_alignment_fallback_success_count"] = float(
                fallback_success_count
            )
        return ([p for p in positions if p is not None], metrics)

    def _build_inserted_response_from_spans(
        self,
        response_text: str,
        response_token_ids: List[int],
        pause_group: List[int],
        collect_metrics: bool = True,
    ) -> Optional[
        Tuple[
            List[int],
            List[int],
            List[bool],
            List[bool],
            List[bool],
            int,
            Dict[str, float],
        ]
    ]:
        visible_response_ids = [int(x) for x in response_token_ids]
        if not visible_response_ids:
            return None
        spans = (
            split_atomic_reasoning_step_spans(response_text) if response_text else []
        )
        span_starts = [int(start) for (start, end) in spans if int(start) < int(end)]
        metrics: Dict[str, float] = {}
        if collect_metrics:
            metrics = {
                "pause/span_count": float(len(spans)),
                "pause/raw_insertion_boundary_count": 0.0,
                "pause/collapsed_insertion_boundary_count": 0.0,
                "pause/unusable_insertion_boundary_count": 0.0,
            }
        if span_starts:
            (raw_positions, map_metrics) = self._map_step_starts_to_token_indices(
                response_text=response_text,
                response_token_ids=visible_response_ids,
                span_starts=span_starts,
                collect_metrics=collect_metrics,
            )
            if collect_metrics:
                metrics.update(map_metrics)
        else:
            raw_positions = [0]
        usable_positions = [
            int(p)
            for p in raw_positions
            if p is not None and 0 <= int(p) < len(visible_response_ids)
        ]
        if collect_metrics:
            metrics["pause/raw_insertion_boundary_count"] = float(len(raw_positions))
            metrics["pause/unusable_insertion_boundary_count"] = float(
                len(raw_positions) - len(usable_positions)
            )
        if not usable_positions:
            usable_positions = [0]
        insertion_positions = sorted(set(usable_positions))
        if collect_metrics:
            metrics["pause/collapsed_insertion_boundary_count"] = float(
                len(usable_positions) - len(insertion_positions)
            )
        insertion_position_set = set(insertion_positions)
        inserted_response_ids: List[int] = []
        inserted_is_pause: List[bool] = []
        inserted_is_boundary: List[bool] = []
        visible_is_boundary: List[bool] = []
        for (token_idx, token_id) in enumerate(visible_response_ids):
            is_boundary = token_idx in insertion_position_set
            if token_idx in insertion_position_set:
                inserted_response_ids.extend(pause_group)
                inserted_is_pause.extend([True] * len(pause_group))
                inserted_is_boundary.extend([False] * len(pause_group))
            inserted_response_ids.append(int(token_id))
            inserted_is_pause.append(False)
            inserted_is_boundary.append(bool(is_boundary))
            visible_is_boundary.append(bool(is_boundary))
        visible_from_inserted = [
            token_id
            for (token_id, is_pause) in zip(inserted_response_ids, inserted_is_pause)
            if not is_pause
        ]
        if visible_from_inserted != visible_response_ids:
            raise RuntimeError(
                f"Pause insertion corrupted visible vLLM token trajectory: visible_from_inserted_len={len(visible_from_inserted)}, visible_response_ids_len={len(visible_response_ids)}"
            )
        return (
            inserted_response_ids,
            visible_response_ids,
            inserted_is_pause,
            inserted_is_boundary,
            visible_is_boundary,
            len(insertion_positions),
            metrics,
        )

    def _build_pause_inserted_sample(
        self,
        prompt: str,
        label: str,
        prompt_token_ids: List[int],
        response_token_ids: List[int],
        response_text: str,
        pause_group: List[int],
        max_response_length: int,
        truncate_length: int,
    ) -> Optional[Experience]:
        prompt_len = len(prompt_token_ids)
        built = self._build_inserted_response_from_spans(
            response_text=response_text,
            response_token_ids=response_token_ids,
            pause_group=pause_group,
            collect_metrics=False,
        )
        if built is None:
            return None
        (
            inserted_response_ids,
            visible_response_ids,
            inserted_is_pause,
            inserted_is_boundary,
            visible_is_boundary,
            step_count,
            pause_metrics,
        ) = built
        input_ids = list(prompt_token_ids) + inserted_response_ids
        if len(input_ids) > truncate_length:
            return None
        sequences = torch.tensor(input_ids, dtype=torch.long, device="cpu")
        attention_mask = torch.ones_like(sequences, dtype=torch.long, device="cpu")
        full_action_mask = torch.tensor(
            [False] * prompt_len + [not is_pause for is_pause in inserted_is_pause],
            dtype=torch.bool,
            device="cpu",
        )
        action_mask = full_action_mask[1:].to("cpu")
        visible_action_count = int(action_mask.sum().item())
        if visible_action_count == 0:
            return None
        ref_ids = list(prompt_token_ids) + list(visible_response_ids)
        ref_full_action_mask = torch.tensor(
            [False] * len(prompt_token_ids) + [True] * len(visible_response_ids),
            dtype=torch.bool,
            device="cpu",
        )
        ref_sequences = torch.tensor(ref_ids, dtype=torch.long, device="cpu")
        ref_attention_mask = torch.ones_like(
            ref_sequences, dtype=torch.long, device="cpu"
        )
        ref_action_mask = ref_full_action_mask[1:].to("cpu")
        original_response_len = len(visible_response_ids)
        total_length_val = int(attention_mask.float().sum().item())
        is_clipped_val = bool(len(response_token_ids) >= max_response_length)
        info = {
            "response_length": torch.tensor(
                [original_response_len], dtype=torch.long, device="cpu"
            ),
            "total_length": torch.tensor(
                [total_length_val], dtype=torch.long, device="cpu"
            ),
            "response_clip_ratio": torch.tensor(
                [is_clipped_val], dtype=torch.bool, device="cpu"
            ),
            "pause_ref_sequences": ref_sequences.unsqueeze(0),
            "pause_ref_attention_mask": ref_attention_mask.unsqueeze(0),
            "pause_ref_action_mask": ref_action_mask.unsqueeze(0),
        }
        rollout_sample = Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=None,
            prompts=[prompt],
            labels=[label],
            info=info,
        )
        return rollout_sample

    def _build_posthoc_no_pause_sample(
        self,
        prompt: str,
        label: str,
        prompt_token_ids: List[int],
        response_token_ids: List[int],
        max_response_length: int,
        truncate_length: int,
    ) -> Optional[Experience]:
        prompt_len = len(prompt_token_ids)
        visible_response_ids = [int(x) for x in response_token_ids]
        if not visible_response_ids:
            return None
        input_ids = list(prompt_token_ids) + visible_response_ids
        if len(input_ids) > truncate_length:
            return None
        sequences = torch.tensor(input_ids, dtype=torch.long, device="cpu")
        attention_mask = torch.ones_like(sequences, dtype=torch.long, device="cpu")
        full_action_mask = torch.tensor(
            [False] * prompt_len + [True] * len(visible_response_ids),
            dtype=torch.bool,
            device="cpu",
        )
        action_mask = full_action_mask[1:].to("cpu")
        visible_action_count = int(action_mask.sum().item())
        if visible_action_count == 0:
            return None
        ref_sequences = sequences.clone()
        ref_attention_mask = attention_mask.clone()
        ref_action_mask = action_mask.clone()
        response_len = len(visible_response_ids)
        total_length_val = int(attention_mask.float().sum().item())
        is_clipped_val = bool(len(response_token_ids) >= max_response_length)
        info = {
            "response_length": torch.tensor(
                [response_len], dtype=torch.long, device="cpu"
            ),
            "total_length": torch.tensor(
                [total_length_val], dtype=torch.long, device="cpu"
            ),
            "response_clip_ratio": torch.tensor(
                [is_clipped_val], dtype=torch.bool, device="cpu"
            ),
            "pause_ref_sequences": ref_sequences.unsqueeze(0),
            "pause_ref_attention_mask": ref_attention_mask.unsqueeze(0),
            "pause_ref_action_mask": ref_action_mask.unsqueeze(0),
        }
        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=None,
            prompts=[prompt],
            labels=[label],
            info=info,
        )

    def _add_functional_rewards(
        self,
        samples_list: List[Experience],
        original_outputs: List[str],
        kwargs: Dict[str, Any],
    ) -> bool:
        args = self.strategy.args
        from evaluation.scripts.eval_utils import math_equal
        from evaluation.scripts.utils import extract_all_answers

        def is_answer_equal(p_ans, ans):
            if p_ans is None or ans is None:
                return False
            return str(p_ans).strip() == str(ans).strip()

        for (i, sample) in enumerate(samples_list):
            full_text = original_outputs[i]
            label = sample.labels[0] if sample.labels else None
            reward_val = 0.0
            if label is not None:
                predicted_answers = extract_all_answers(full_text, args.answer_trigger)
                if predicted_answers is not None:
                    for p_ans in predicted_answers:
                        if p_ans is not None and (
                            is_answer_equal(p_ans, label) or math_equal(p_ans, label)
                        ):
                            reward_val = 1.0
                            break
            reward_tensor = torch.tensor(
                [reward_val], dtype=torch.float32, device="cpu"
            )
            sample.rewards = reward_tensor.clone()
            sample.scores = reward_tensor.clone()
            sample.info["reward"] = reward_tensor.clone()
            sample.info["score"] = reward_tensor.clone()
        return True

    def _finalize_pause_rollout_samples(
        self,
        samples_list: List[Experience],
        original_outputs: List[str],
        kwargs: Dict[str, Any],
    ) -> List[Experience]:
        if not samples_list:
            return []
        self._add_functional_rewards(samples_list, original_outputs, kwargs)
        self._assert_same_info_keys(samples_list)
        return samples_list

    def _build_posthoc_sampling_params(
        self,
        *,
        max_response_length: int,
        pause_group: List[int],
        kwargs: Dict[str, Any],
    ):
        from vllm import SamplingParams

        args = self.strategy.args
        pause_id = int(pause_group[0])
        sampling_kwargs = dict(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", -1),
            max_tokens=max_response_length,
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05),
            include_stop_str_in_output=True,
            logprobs=None,
            bad_words=[args.pause_token],
            logit_bias={pause_id: -100.0},
        )
        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Post-rollout pause GRPO requires SamplingParams with bad_words and logit_bias support. Install the pinned training environment; the pause-token sampling ban cannot be disabled."
            ) from exc
        return sampling_params

    def _generate_vllm_posthoc_expanded(
        self,
        all_prompts: List[str],
        all_labels,
        all_prompt_token_ids: List[List[int]],
        *,
        max_response_length: int,
        truncate_length: int,
        pause_group: List[int],
        kwargs: Dict[str, Any],
    ) -> Tuple[List[Optional[Experience]], List[str]]:
        llms = self.vllm_engines
        if not llms:
            raise ValueError("Pause GRPO requires vLLM engines for rollout generation.")
        if not all_prompts:
            return ([], [])
        sampling_params = self._build_posthoc_sampling_params(
            max_response_length=max_response_length,
            pause_group=pause_group,
            kwargs=kwargs,
        )
        refs = []
        active_llms = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for (i, llm) in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[
                i * batch_size : (i + 1) * batch_size
            ]
            if prompt_token_ids:
                refs.append(
                    llm.add_requests.remote(
                        sampling_params=sampling_params,
                        prompt_token_ids=prompt_token_ids,
                    )
                )
                active_llms.append(llm)
        ray.get(refs)
        all_output_refs = [llm.get_responses.remote() for llm in active_llms]
        all_outputs = sum(ray.get(all_output_refs), [])
        if len(all_outputs) != len(all_prompt_token_ids):
            raise RuntimeError(
                f"Pause GRPO posthoc rollout count mismatch: expected={len(all_prompt_token_ids)}, got={len(all_outputs)}"
            )
        global_step = int(kwargs.get("global_step", 0) or 0)
        (posthoc_ratio, pause_routes) = self._posthoc_pause_routes_for_batch(
            len(all_outputs), global_step
        )
        applied_sample_count = sum((int(use_pause) for use_pause in pause_routes))
        skipped_sample_count = len(pause_routes) - applied_sample_count
        sample_slots = []
        original_outputs = []
        for (i, output) in enumerate(all_outputs):
            prompt = all_prompts[i]
            label = all_labels[i]
            completion = output.outputs[0]
            original_text = completion.text
            original_response_token_ids = list(completion.token_ids)
            use_pause = bool(pause_routes[i])
            if use_pause:
                sample = self._build_pause_inserted_sample(
                    prompt=prompt,
                    label=label,
                    prompt_token_ids=list(output.prompt_token_ids),
                    response_token_ids=original_response_token_ids,
                    response_text=original_text,
                    pause_group=pause_group,
                    max_response_length=max_response_length,
                    truncate_length=truncate_length,
                )
            else:
                sample = self._build_posthoc_no_pause_sample(
                    prompt=prompt,
                    label=label,
                    prompt_token_ids=list(output.prompt_token_ids),
                    response_token_ids=original_response_token_ids,
                    max_response_length=max_response_length,
                    truncate_length=truncate_length,
                )
            if sample is not None:
                self._stamp_posthoc_pause_ratio_metrics(
                    sample,
                    pause_applied=use_pause,
                    posthoc_ratio=posthoc_ratio,
                    applied_sample_count=applied_sample_count,
                    skipped_sample_count=skipped_sample_count,
                )
            sample_slots.append(sample)
            original_outputs.append(original_text)
        return (sample_slots, original_outputs)

    def _generate_vllm_posthoc(
        self, all_prompts: List[str], all_labels, **kwargs
    ) -> List[Experience]:
        args = self.strategy.args
        max_response_length = kwargs.get("max_new_tokens", 2048)
        truncate_length = self.prompt_max_len + max_response_length
        pause_group = self._pause_token_ids()
        n_samples_per_prompt = kwargs.pop(
            "n_samples_per_prompt", args.n_samples_per_prompt
        )
        all_prompts = sum(
            [[prompt] * n_samples_per_prompt for prompt in all_prompts], []
        )
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False
        )["input_ids"]
        (sample_slots, original_outputs) = self._generate_vllm_posthoc_expanded(
            all_prompts,
            all_labels,
            all_prompt_token_ids,
            max_response_length=max_response_length,
            truncate_length=truncate_length,
            pause_group=pause_group,
            kwargs=kwargs,
        )
        (
            samples_list,
            original_outputs,
            dropped_groups,
        ) = self._filter_complete_pause_groups(
            sample_slots, original_outputs, n_samples_per_prompt
        )
        for sample in samples_list:
            pass
        if samples_list and original_outputs:
            logger.info(
                f"🚀 Pause GRPO original rollout sample: {samples_list[0].prompts[0] + original_outputs[0]}"
            )
        if dropped_groups > 0:
            logger.warning(
                f"Pause GRPO dropped {dropped_groups} prompt group(s) because at least one sample was invalid."
            )
        return self._finalize_pause_rollout_samples(
            samples_list, original_outputs, kwargs
        )

    def _generate_vllm(
        self, all_prompts: List[str], all_labels, **kwargs
    ) -> List[Experience]:
        llms = self.vllm_engines
        args = self.strategy.args
        if not llms:
            raise ValueError("Pause GRPO requires vLLM engines for rollout generation.")
        if getattr(args, "enable_vllm_is_correction", False):
            raise ValueError("Pause GRPO does not support --enable_vllm_is_correction.")
        return self._generate_vllm_posthoc(all_prompts, all_labels, **kwargs)


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__()
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args
        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        for (i, sample) in enumerate(rollout_samples):
            sample.index = [i]
        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [
                int(s.info["total_length"].item()) for s in rollout_samples
            ]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = (
                minimum_batch_num // effective_actor_num * effective_actor_num
            )
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(
                total_lengths, num_batch, False
            )
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(
                    micro_batch, self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(
        self, rollout_samples, global_step: Optional[int] = None
    ) -> List[Experience]:
        samples_list = self.split_rollout_samples(rollout_samples)
        experiences = self.make_experience(samples_list)
        return self.compute_advantages_and_returns(experiences, global_step=global_step)

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(
            f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples"
        )
        args = self.strategy.args
        device = "cpu"
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        use_lora_disabled_ref_model = (
            getattr(args, "use_lora_disabled_ref_model", False)
            and args.init_kl_coef > 0
        )
        if any((sample.rewards is None for sample in samples_list)):
            raise ValueError(
                "GRPO requires mathematical rewards from the rollout generator."
            )
        actor_kwargs = {
            "sequences": sequences_list,
            "action_mask": action_mask_list,
            "attention_mask": attention_mask_list,
        }
        if use_lora_disabled_ref_model:
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward_with_lora_disabled_ref", **actor_kwargs
            )
        else:
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward", **actor_kwargs
            )
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))
        if self.critic_model_group is not None:
            if args.advantage_estimator in ["group_norm", "dr_grpo"]:
                raise ValueError(
                    f"Advantage estimator {args.advantage_estimator} not use critic model"
                )
            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(
                    self.critic_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            value_ref = ray.put(
                [[None]]
                * (
                    len(samples_list)
                    * args.ring_attn_size
                    * args.ds_tensor_parallel_size
                )
            )
        if use_lora_disabled_ref_model:
            base_action_log_probs_ref = None
        elif self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(
                    self.initial_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            base_action_log_probs_ref = ray.put(
                [[None]]
                * (
                    len(samples_list)
                    * args.ring_attn_size
                    * args.ds_tensor_parallel_size
                )
            )
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        if use_lora_disabled_ref_model:
            actor_log_probs_list = sum(
                ray.get(action_log_probs_ref)[::duplicate_factor], []
            )
            action_log_probs_list = [item[0] for item in actor_log_probs_list]
            base_action_log_probs_list = [item[1] for item in actor_log_probs_list]
        else:
            action_log_probs_list = sum(
                ray.get(action_log_probs_ref)[::duplicate_factor], []
            )
            base_action_log_probs_list = sum(
                ray.get(base_action_log_probs_ref)[::duplicate_factor], []
            )
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        assert (
            len(samples_list)
            == len(action_log_probs_list)
            == len(base_action_log_probs_list)
            == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"
        has_reference_log_probs = (
            self.initial_model_group is not None or use_lora_disabled_ref_model
        )
        for (i, (samples, action_log_probs, base_action_log_probs, value)) in enumerate(
            zip(
                samples_list,
                action_log_probs_list,
                base_action_log_probs_list,
                value_list,
            )
        ):
            if has_reference_log_probs and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                kl_for_reward = kl
            else:
                kl = torch.zeros_like(
                    action_log_probs, dtype=action_log_probs.dtype, device=device
                )
                kl_for_reward = kl
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            if not args.use_kl_loss:
                base_action_log_probs = None
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl_for_reward
            samples.info["kl"] = kl_mean
        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        global_step = kwargs.get("global_step", None)
        step = 0 if global_step is None else max(0, int(global_step))
        pause_penalty_warmup_steps = max(
            0, int(getattr(args, "pause_penalty_warmup_steps", 0))
        )
        pause_penalty_ramp_steps = max(
            0, int(getattr(args, "pause_penalty_ramp_steps", 0))
        )
        if step < pause_penalty_warmup_steps:
            pause_penalty_scale = 0.0
        elif pause_penalty_ramp_steps <= 0:
            pause_penalty_scale = 1.0
        else:
            pause_penalty_scale = min(
                1.0, (step - pause_penalty_warmup_steps) / pause_penalty_ramp_steps
            )
        if args.overlong_buffer_len is not None:
            assert (
                args.generate_max_len >= args.overlong_buffer_len
            ), "generate_max_len must be larger than overlong_buffer_len"
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    exceed_len = min(
                        valid_response_length - expected_len, overlong_buffer_len
                    )
                    if exceed_len > 0:
                        overlong_penalty = (
                            -exceed_len / overlong_buffer_len * overlong_penalty_factor
                        )
                        experience.rewards[j] += overlong_penalty
        exp_len = [len(experience.index) for experience in experiences]
        indices = torch.tensor(
            sum([experience.index for experience in experiences], [])
        )
        raw_rewards = torch.cat(
            [experience.rewards for experience in experiences], dim=0
        )
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True)
                .repeat(1, args.n_samples_per_prompt)
                .reshape(-1)[indices]
                .split(exp_len)
            )
            for (experience, group_reward_std) in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (
                args.n_samples_per_prompt - 1
            )
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (
                rewards.std(-1, keepdim=True) + 1e-09
            )
        rewards = rewards.reshape(-1)[indices].split(exp_len)
        for (experience, reward) in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )
            if getattr(args, "grpo_mode", None) == "pause":
                action_mask = experience.action_mask.to(
                    device=reward.device, dtype=torch.bool
                )
                target_len = int(getattr(args, "pause_length_target", 0))
                base_clip_penalty = float(getattr(args, "pause_clip_penalty", 0.0))
                base_token_penalty = float(
                    getattr(args, "pause_over_target_token_penalty", 0.0)
                )
                effective_clip_penalty = pause_penalty_scale * base_clip_penalty
                effective_token_penalty = pause_penalty_scale * base_token_penalty
                clip_penalties = torch.zeros(
                    reward.size(0), device=reward.device, dtype=reward.dtype
                )
                length_penalties = torch.zeros_like(clip_penalties)
                over_target_lengths = torch.zeros_like(clip_penalties)
                if target_len > 0:
                    response_lengths = (
                        experience.info["response_length"]
                        .to(device=reward.device, dtype=reward.dtype)
                        .view(-1)
                    )
                    over_target_lengths = (response_lengths - target_len).clamp(min=0)
                if effective_clip_penalty > 0:
                    clipped = (
                        experience.info["response_clip_ratio"]
                        .to(device=reward.device, dtype=reward.dtype)
                        .view(-1)
                    )
                    clip_penalties = effective_clip_penalty * clipped
                if target_len > 0 and effective_token_penalty > 0:
                    length_penalties = effective_token_penalty * over_target_lengths
                correct_gate = torch.ones_like(clip_penalties)
                if getattr(args, "pause_penalty_correct_only", False):
                    raw_score = experience.scores
                    if raw_score is None:
                        raw_score = experience.info.get("score")
                    if raw_score is None:
                        raw_score = experience.rewards
                    correct_gate = (
                        raw_score.to(device=reward.device, dtype=reward.dtype).view(-1)
                        > 0.5
                    ).to(dtype=reward.dtype)
                    clip_penalties = clip_penalties * correct_gate
                    length_penalties = length_penalties * correct_gate
                total_penalties = clip_penalties + length_penalties
                penalty_active = (total_penalties > 0).to(dtype=reward.dtype)
                has_action = action_mask.any(dim=-1)
                if has_action.any():
                    last_action_idx = (
                        action_mask.size(1)
                        - 1
                        - action_mask.long().fliplr().argmax(dim=1)
                    )
                    batch_idx = torch.arange(reward.size(0), device=reward.device)
                    reward[
                        batch_idx[has_action], last_action_idx[has_action]
                    ] -= total_penalties[has_action]
                experience.info["pause/penalty_scale"] = torch.full_like(
                    total_penalties.detach().cpu(), float(pause_penalty_scale)
                )
                experience.info["pause/clip_penalty"] = clip_penalties.detach().cpu()
                experience.info[
                    "pause/over_target_length"
                ] = over_target_lengths.detach().cpu()
                experience.info[
                    "pause/over_target_length_penalty"
                ] = length_penalties.detach().cpu()
                experience.info[
                    "pause/total_length_penalty"
                ] = total_penalties.detach().cpu()
                experience.info[
                    "pause/penalty_correct_gate"
                ] = correct_gate.detach().cpu()
                experience.info["pause/penalty_active"] = penalty_active.detach().cpu()
                experience.info["pause/effective_clip_penalty_coef"] = torch.full_like(
                    total_penalties.detach().cpu(), float(effective_clip_penalty)
                )
                experience.info[
                    "pause/effective_over_target_token_penalty_coef"
                ] = torch.full_like(
                    total_penalties.detach().cpu(), float(effective_token_penalty)
                )
            if self.advantage_estimator == "gae":
                (
                    experience.advantages,
                    experience.returns,
                ) = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in [
                "reinforce",
                "rloo",
                "reinforce_baseline",
                "group_norm",
                "dr_grpo",
            ]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning(
                        "gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm"
                    )
                    args.gamma = 1.0
                experience.returns = self.get_cumulative_returns(
                    reward, experience.action_mask, args.gamma
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(
                    f"Unkown advantage_estimator {self.advantage_estimator}"
                )
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            experience.kl = None
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())
            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            if not self.args.no_advantage_std_norm:
                var = (
                    (advantages_vector - mean).pow(2) * action_masks_vector
                ).sum() / num_actions
                rstd = var.clamp(min=1e-08).rsqrt()
            else:
                rstd = 1
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd
        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return (advantages.detach(), returns)

    @torch.no_grad()
    def get_cumulative_returns(
        self, rewards: torch.Tensor, action_mask: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)
        if action_mask is not None:
            rewards = action_mask * rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return
        return returns


class PauseGRPORemoteExperienceMaker(RemoteExperienceMaker):
    """Experience maker for pause-inserted GRPO rollouts."""

    @staticmethod
    def _align_original_reference_log_probs(
        samples: Experience, ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        current_action_mask = samples.action_mask.bool()
        ref_action_mask = samples.info["pause_ref_action_mask"].bool()
        current_targets = samples.sequences[:, 1:]
        ref_targets = samples.info["pause_ref_sequences"][:, 1:]
        aligned = torch.zeros(
            current_action_mask.shape,
            dtype=ref_log_probs.dtype,
            device=ref_log_probs.device,
        )
        for row_idx in range(current_action_mask.size(0)):
            current_indices = torch.nonzero(
                current_action_mask[row_idx], as_tuple=False
            ).flatten()
            current_ids = current_targets[row_idx][current_action_mask[row_idx]]
            ref_ids = ref_targets[row_idx][ref_action_mask[row_idx]]
            if current_ids.numel() != ref_ids.numel() or not torch.equal(
                current_ids.cpu(), ref_ids.cpu()
            ):
                raise RuntimeError(
                    f"Original-reference pause KL token alignment mismatch: row={row_idx}, current_visible_targets={current_ids.numel()}, reference_visible_targets={ref_ids.numel()}, current_prefix={current_ids[:32].cpu().tolist()}, reference_prefix={ref_ids[:32].cpu().tolist()}"
                )
            ref_values = ref_log_probs[row_idx][
                ref_action_mask[row_idx].to(ref_log_probs.device)
            ]
            if current_indices.numel() != ref_values.numel():
                raise RuntimeError(
                    f"Original-reference pause KL logprob alignment mismatch: current visible targets={current_indices.numel()}, reference visible logprobs={ref_values.numel()}"
                )
            if current_indices.numel() > 0:
                aligned[row_idx, current_indices.to(aligned.device)] = ref_values
        return aligned

    @staticmethod
    def _drop_pause_reference_metadata(samples: Experience) -> None:
        for key in (
            "pause_ref_sequences",
            "pause_ref_attention_mask",
            "pause_ref_action_mask",
        ):
            samples.info.pop(key, None)

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        start_time = time.time()
        logger.info(
            f"🚀 Starting pause experience making with {sum([len(s.sequences) for s in samples_list])} samples"
        )
        args = self.strategy.args
        device = "cpu"
        reference_mode = getattr(args, "pause_reference_mode", "original")
        if reference_mode not in {"original", "inserted"}:
            raise ValueError(f"Unknown pause_reference_mode: {reference_mode}")
        if not samples_list:
            return []
        if any((samples.rewards is None for samples in samples_list)):
            raise ValueError(
                "PauseGRPO reward must be computed from original no-pause rollout text before experience making"
            )
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        use_lora_disabled_ref_model = (
            getattr(args, "use_lora_disabled_ref_model", False)
            and args.init_kl_coef > 0
        )
        actor_kwargs = {
            "sequences": sequences_list,
            "action_mask": action_mask_list,
            "attention_mask": attention_mask_list,
        }
        if use_lora_disabled_ref_model:
            if reference_mode == "original":
                ref_sequences_list = [
                    s.info["pause_ref_sequences"].long() for s in samples_list
                ]
                ref_attention_mask_list = [
                    s.info["pause_ref_attention_mask"].long() for s in samples_list
                ]
                ref_action_mask_list = [
                    s.info["pause_ref_action_mask"].bool() for s in samples_list
                ]
            else:
                ref_sequences_list = [None] * len(samples_list)
                ref_attention_mask_list = [None] * len(samples_list)
                ref_action_mask_list = [None] * len(samples_list)
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward_with_lora_disabled_ref",
                **actor_kwargs,
                ref_sequences=ref_sequences_list,
                ref_action_mask=ref_action_mask_list,
                ref_attention_mask=ref_attention_mask_list,
            )
        else:
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward", **actor_kwargs
            )
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))
        if self.critic_model_group is not None:
            if args.advantage_estimator in ["group_norm", "dr_grpo"]:
                raise ValueError(
                    f"Advantage estimator {args.advantage_estimator} not use critic model"
                )
            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(
                    self.critic_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            value_ref = ray.put(
                [[None]]
                * (
                    len(samples_list)
                    * args.ring_attn_size
                    * args.ds_tensor_parallel_size
                )
            )
        if use_lora_disabled_ref_model:
            base_action_log_probs_ref = None
        elif self.initial_model_group is not None:
            if reference_mode == "original":
                ref_sequences_list = [
                    s.info["pause_ref_sequences"].long() for s in samples_list
                ]
                ref_attention_mask_list = [
                    s.info["pause_ref_attention_mask"].long() for s in samples_list
                ]
                ref_action_mask_list = [
                    s.info["pause_ref_action_mask"].bool() for s in samples_list
                ]
                base_action_log_probs_ref = (
                    self.initial_model_group.async_run_method_batch(
                        method_name="forward",
                        sequences=ref_sequences_list,
                        action_mask=ref_action_mask_list,
                        attention_mask=ref_attention_mask_list,
                    )
                )
            else:
                base_action_log_probs_ref = (
                    self.initial_model_group.async_run_method_batch(
                        method_name="forward",
                        sequences=sequences_list,
                        action_mask=action_mask_list,
                        attention_mask=attention_mask_list,
                    )
                )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(
                    self.initial_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            base_action_log_probs_ref = ray.put(
                [[None]]
                * (
                    len(samples_list)
                    * args.ring_attn_size
                    * args.ds_tensor_parallel_size
                )
            )
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        if use_lora_disabled_ref_model:
            actor_log_probs_list = sum(
                ray.get(action_log_probs_ref)[::duplicate_factor], []
            )
            action_log_probs_list = [item[0] for item in actor_log_probs_list]
            base_action_log_probs_list = [item[1] for item in actor_log_probs_list]
        else:
            action_log_probs_list = sum(
                ray.get(action_log_probs_ref)[::duplicate_factor], []
            )
            base_action_log_probs_list = sum(
                ray.get(base_action_log_probs_ref)[::duplicate_factor], []
            )
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        assert (
            len(samples_list)
            == len(action_log_probs_list)
            == len(base_action_log_probs_list)
            == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"
        has_reference_log_probs = (
            self.initial_model_group is not None or use_lora_disabled_ref_model
        )
        for (samples, action_log_probs, base_action_log_probs, value) in zip(
            samples_list, action_log_probs_list, base_action_log_probs_list, value_list
        ):
            if base_action_log_probs is not None and reference_mode == "original":
                base_action_log_probs = self._align_original_reference_log_probs(
                    samples, base_action_log_probs
                )
            if has_reference_log_probs and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                kl_for_reward = kl
            else:
                kl = torch.zeros_like(
                    action_log_probs, dtype=action_log_probs.dtype, device=device
                )
                kl_for_reward = kl
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            if not args.use_kl_loss:
                base_action_log_probs = None
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl_for_reward
            samples.info["kl"] = kl_mean
            self._drop_pause_reference_metadata(samples)
        end_time = time.time()
        time_str = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        logger.info(f"✨ Pause experience making completed in {time_str}")
        return samples_list
