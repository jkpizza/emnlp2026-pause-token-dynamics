from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch



def extract_first_numeric_answer(text):
    matches = []

    # pattern 1: \(\\boxed{ANSWER}\)
    match1 = re.search(r'\\\(\\boxed\{(.*?)\}\\\)', text)
    if match1:
        matches.append(('boxed', match1.start(), match1.group(1).strip()))

    # pattern 2: Therefore, the answer is: ANSWER.
    match2 = re.search(r'Therefore, the answer is: ([^\.\n]+)', text)
    if match2:
        matches.append(('therefore', match2.start(), match2.group(1).strip()))

    if not matches:
        return None

    # extract first answer
    first = min(matches, key=lambda x: x[1])
    answer_text = first[2]

    # float / int extract
    num_match = re.search(r'\d+(?:\.\d+)?', answer_text)
    return float(num_match.group()) if num_match else None


## match answer include non-numeric answer

def extract_last_answer(text: str, answer_trigger: str):
    """
    Extracts the last expression following the answer_trigger in the text.
    """
    # 2. Fallback to \\boxed{...} with nested {} support
    # Match \boxed{...} where ... may include nested {...}
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    if matches_boxed:
        # Unwrap double braces if present (e.g., {{...}})
        boxed_content = matches_boxed[-1].strip()
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        return boxed_content

    pattern = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches = re.findall(pattern, text)
    if matches:
        last_answer = matches[-1].strip()
        if last_answer:
            return last_answer


    # Nothing found
    return None
import re

def extract_all_answers(text: str, answer_trigger: str):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # 2. Find all \boxed{...} matches with nested {} support
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    for boxed_content in matches_boxed:
        boxed_content = boxed_content.strip()
        # Unwrap double braces if present (e.g., {{...}})
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        if boxed_content:
            results.append(boxed_content)

    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None

def extract_all_answers_not_boxed(text: str, answer_trigger: str):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # # 2. Find all \boxed{...} matches with nested {} support
    # pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    # matches_boxed = re.findall(pattern_boxed, text)
    # for boxed_content in matches_boxed:
    #     boxed_content = boxed_content.strip()
    #     # Unwrap double braces if present (e.g., {{...}})
    #     if boxed_content.startswith('{') and boxed_content.endswith('}'):
    #         boxed_content = boxed_content[1:-1].strip()
    #     if boxed_content:
    #         results.append(boxed_content)

    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None

def normalize_expression(expr: str) -> str:
    """
    Normalize expressions by removing whitespace and wrapping quotes.
    """
    if expr is None:
        return None
    return re.sub(r'\s+', '', expr).strip("'\"")


def try_parse_float(expr: str):
    """
    Tries to convert an expression to float. Returns None if not possible.
    """
    try:
        return float(expr)
    except ValueError:
        return None


def is_answer_equal(predicted: str, target: str) -> bool:
    import math
    """
    Compares predicted and target answers:
    - If both are numeric, compare numerically
    - Otherwise, compare after normalizing expressions
    """
    pred_norm = normalize_expression(predicted)
    target_norm = normalize_expression(target)

    pred_num = try_parse_float(pred_norm)
    target_num = try_parse_float(target_norm)

    if pred_num is not None and target_num is not None:
        return math.isclose(pred_num, target_num, rel_tol=1e-6)

    return pred_norm == target_norm


def match_with_answer_labels_v2(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                predicted_answer = extract_last_answer(output, answer_trigger)
                if predicted_answer is None:
                    continue
                is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1

        return correct_count/valid_count if valid_count > 0 else 0

def match_with_answer_labels_v3(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        extracted_answer = None
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                # predicted_answer = extract_last_answer(output, answer_trigger)
                predicted_answers = extract_all_answers(output, answer_trigger)
                if predicted_answers is None:
                    continue
                is_correct = False
                # Check if any of the predicted answers match the target answer
                for predicted_answer in predicted_answers:
                    if predicted_answer is None:
                        continue
                    if is_answer_equal(predicted_answer, answer):
                        is_correct = True
                        extracted_answer = predicted_answer
                        break
                # is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1

        accuracy = correct_count/valid_count if valid_count > 0 else 0
        return accuracy


def match_with_answer_labels_v4(tokenized_output, answers, answer_trigger):
    from evaluation_scripts.evaluation.eval.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct


    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy


def match_with_answer_labels_not_boxed(tokenized_output, answers, answer_trigger):
    from evaluation_scripts.evaluation.eval.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers_not_boxed(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct


    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy
