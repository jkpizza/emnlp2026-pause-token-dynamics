from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from tqdm import tqdm
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

from reproduction.data import load_source, validate_columns
from openrlhf.utils.code_pause import code_line_starts, insert_at_token_boundaries, insert_pauses_before_code_lines


PAUSE_TOKEN = "<|PAUSE|>"
METHODS = ("none", "append", "random", "dit", "step", "code_line")
BUDGET_MODES = ("exact_steps", "avg_steps")
SPLITTER_REGISTRY = {
    "atomic_math": "openrlhf.utils.split_steps:split_atomic_reasoning_steps",
}


def unicode_range(start: int, end: int) -> str:
    return "".join(chr(c) for c in range(start, end + 1))


VALID_LATEXY = re.compile(
    r"\A["
    + r"A-Za-z0-9"
    + r"\s"
    + r"\.\,\;\:\!\?\'\"\_\-\+\=\*\/\^\|\%\<\>\~\#\@\&\$"
    + re.escape("\\()[]{}")
    + unicode_range(0x0370, 0x03FF)
    + unicode_range(0x1F00, 0x1FFF)
    + unicode_range(0x2190, 0x21FF)
    + unicode_range(0x2200, 0x22FF)
    + unicode_range(0x27C0, 0x27EF)
    + unicode_range(0x2980, 0x29FF)
    + unicode_range(0x2A00, 0x2AFF)
    + unicode_range(0x2100, 0x214F)
    + unicode_range(0x2070, 0x209F)
    + unicode_range(0x1D400, 0x1D7FF)
    + "\u00B0\u00B1\u00B2\u00B3\u00B7\u00B9\u00D7\u00F7"
    + r"]*\Z",
    re.UNICODE,
)


@dataclass
class RawSample:
    prompt: str
    response: str
    step_count: int
    steps: list[str]
    metadata: dict


def is_valid_generated_text(text: str) -> bool:
    return bool(VALID_LATEXY.match(text))


def pick_invalid_tokens(vocab: dict[str, int], need: int) -> list[str]:
    items = [(tok, tid, len(tok)) for tok, tid in vocab.items()]
    items.sort(key=lambda x: (-x[2], x[1]))
    out = []
    for tok, _, _ in items:
        if not is_valid_generated_text(tok):
            out.append(tok)
            if len(out) >= need:
                return out
    raise ValueError(f"Not enough invalid tokenizer tokens to replace; need {need}.")


def one_token_id(tokenizer, token: str) -> Optional[int]:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if isinstance(ids, list) and len(ids) == 1:
        return int(ids[0])
    return None


def rewrite_tokenizer_files_with_pause(save_path: str, pause_token: str) -> None:
    tokjson = os.path.join(save_path, "tokenizer.json")
    tokcfg = os.path.join(save_path, "tokenizer_config.json")
    vocabjson = os.path.join(save_path, "vocab.json")
    mergestxt = os.path.join(save_path, "merges.txt")
    addedtoks = os.path.join(save_path, "added_tokens.json")

    tokdata = None
    vocab = None
    merges_json = None
    if os.path.exists(tokjson):
        with open(tokjson, "r", encoding="utf-8") as f:
            tokdata = json.load(f)
        vocab = tokdata.get("model", {}).get("vocab")
        merges_json = tokdata.get("model", {}).get("merges", [])

    vocab_json_data = None
    if os.path.exists(vocabjson):
        with open(vocabjson, "r", encoding="utf-8") as f:
            vocab_json_data = json.load(f)
        if vocab is None:
            vocab = vocab_json_data

    if not isinstance(vocab, dict):
        raise RuntimeError("Pause tokenizer build expects tokenizer.json model.vocab or vocab.json.")
    if pause_token in vocab:
        return

    old_token = pick_invalid_tokens(vocab, 1)[0]
    token_id = int(vocab.pop(old_token))
    vocab[pause_token] = token_id

    if isinstance(vocab_json_data, dict) and old_token in vocab_json_data:
        token_id_json = int(vocab_json_data.pop(old_token))
        vocab_json_data[pause_token] = token_id_json

    if merges_json is not None:
        kept = []
        for rule in merges_json:
            merged = "".join(rule) if isinstance(rule, list) else str(rule).replace(" ", "")
            if old_token not in merged:
                kept.append(rule)
        merges_json = kept

    if os.path.exists(mergestxt):
        with open(mergestxt, "r", encoding="utf-8") as f:
            merge_lines = f.read().splitlines()
        kept_lines = []
        for line in merge_lines:
            if old_token not in "".join(line.split()):
                kept_lines.append(line)
        with open(mergestxt, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines))

    tokcfg_data = {}
    if os.path.exists(tokcfg):
        with open(tokcfg, "r", encoding="utf-8") as f:
            tokcfg_data = json.load(f)
    decoder = tokcfg_data.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        decoder = {}
    decoder[str(token_id)] = {
        "content": pause_token,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": False,
    }
    tokcfg_data["added_tokens_decoder"] = decoder
    with open(tokcfg, "w", encoding="utf-8") as f:
        json.dump(tokcfg_data, f, ensure_ascii=False, indent=2)

    if tokdata is not None:
        added_list = tokdata.get("added_tokens", [])
        if not isinstance(added_list, list):
            added_list = []
        id_to_idx = {int(e["id"]): i for i, e in enumerate(added_list) if "id" in e}
        payload = {
            "id": token_id,
            "content": pause_token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        if token_id in id_to_idx:
            added_list[id_to_idx[token_id]] = payload
        else:
            added_list.append(payload)
        tokdata["added_tokens"] = added_list
        tokdata["model"]["vocab"] = vocab
        if merges_json is not None:
            tokdata["model"]["merges"] = merges_json
        with open(tokjson, "w", encoding="utf-8") as f:
            json.dump(tokdata, f, ensure_ascii=False, indent=2)

    if vocab_json_data is not None:
        with open(vocabjson, "w", encoding="utf-8") as f:
            json.dump(vocab_json_data, f, ensure_ascii=False, indent=2)

    added_tokens = {}
    if os.path.exists(addedtoks):
        with open(addedtoks, "r", encoding="utf-8") as f:
            added_tokens = json.load(f)
    if isinstance(added_tokens, dict):
        added_tokens[pause_token] = token_id
        with open(addedtoks, "w", encoding="utf-8") as f:
            json.dump(added_tokens, f, ensure_ascii=False, indent=2)

    print(f"Pause tokenizer token replaced: {old_token!r} -> {pause_token!r} at id {token_id}")


def ensure_pause_tokenizer(tokenizer_path: str, save_path: str, pause_token: str, use_fast: bool):
    source = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=use_fast)
    source_id = one_token_id(source, pause_token)
    os.makedirs(save_path, exist_ok=True)
    source.save_pretrained(save_path)
    if source_id is None:
        rewrite_tokenizer_files_with_pause(save_path, pause_token)

    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True, trust_remote_code=False, use_fast=use_fast)
    pause_id = one_token_id(tokenizer, pause_token)
    if pause_id is None:
        ids = tokenizer.encode(pause_token, add_special_tokens=False)
        raise ValueError(f"{pause_token!r} must tokenize to exactly one id, got {ids}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, pause_id


def load_splitter(splitter_name: str) -> Callable[[str], list[str]]:
    spec = SPLITTER_REGISTRY.get(splitter_name, splitter_name)
    if ":" not in spec:
        raise ValueError("Use a registry key like 'atomic_math' or 'module.submodule:function_name'.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def dataset_slug(dataset: str) -> str:
    base = os.path.basename(dataset.rstrip("/")) or dataset.replace("/", "_")
    if os.path.splitext(base)[1] in {".json", ".jsonl", ".csv", ".parquet", ".arrow"}:
        base = os.path.splitext(base)[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base.replace("/", "_")).strip("_") or "dataset"


def split_response(response: str, tokenizer, args, splitter_fn: Callable[[str], list[str]]) -> list[str]:
    if args.split_mode == "chunk":
        ids = tokenizer.encode(response, add_special_tokens=False)
        steps = []
        for start in range(0, len(ids), args.chunk_size):
            chunk = ids[start : start + args.chunk_size]
            if chunk:
                steps.append(tokenizer.decode(chunk, skip_special_tokens=False))
        return steps
    return [x.strip() for x in splitter_fn(response) if x and x.strip()]


def load_raw_samples(dataset, args, tokenizer, splitter_fn) -> list[RawSample]:
    rows = []
    for idx, row in enumerate(tqdm(dataset, desc="Preparing raw samples")):
        prompt = row[args.input_key]
        response = row[args.output_key]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) >= args.max_length - 2:
            continue
        if getattr(args, "domain", "math") == "code":
            try:
                starts = code_line_starts(response)
            except ValueError as exc:
                raise ValueError(f"Code dataset row {idx}: {exc}") from exc
            if not starts:
                raise ValueError(f"Code dataset row {idx} has no nonblank fenced code lines; "
                                 "put the solution inside a ```python ... ``` block.")
            steps, step_count = [], len(starts)
        else:
            steps = split_response(response, tokenizer, args, splitter_fn)
            step_count = max(1, len(steps))
        rows.append(
            RawSample(
                prompt=prompt,
                response=response,
                step_count=step_count,
                steps=steps,
                metadata={"source_index": idx},
            )
        )
    return rows


def pause_ids(pause_id: int, pause_tokens_per_step: int) -> list[int]:
    return [int(pause_id)] * int(pause_tokens_per_step)


def insert_groups(ids: list[int], boundaries: list[int], group_ids: list[int]) -> list[int]:
    by_boundary = {}
    for boundary in boundaries:
        boundary = max(0, min(len(ids), int(boundary)))
        by_boundary[boundary] = by_boundary.get(boundary, 0) + 1

    out = []
    for idx in range(len(ids) + 1):
        if idx in by_boundary:
            for _ in range(by_boundary[idx]):
                out.extend(group_ids)
        if idx < len(ids):
            out.append(ids[idx])
    return out


def random_boundaries(length: int, count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    candidates = list(range(length + 1))
    if count <= len(candidates):
        return sorted(rng.sample(candidates, count))
    values = candidates[:]
    values.extend(rng.choice(candidates) for _ in range(count - len(candidates)))
    return sorted(values)


def uniform_boundaries(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return [max(0, min(length, int(round((i + 1) * length / (count + 1))))) for i in range(count)]


class DitScorer:
    def __init__(self, model_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype or "auto",
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def lowest_logprob_boundaries(self, prompt_ids: list[int], response_ids: list[int], count: int) -> list[int]:
        if count <= 0 or not response_ids:
            return []
        usable_response_len = min(len(response_ids), max(0, self.max_length - len(prompt_ids)))
        if usable_response_len <= 0:
            return uniform_boundaries(len(response_ids), count)
        input_ids = prompt_ids + response_ids[:usable_response_len]
        if len(input_ids) < 2:
            return uniform_boundaries(len(response_ids), count)

        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor).logits[:, :-1, :].float()
        next_tokens = tensor[:, 1:]
        logps = torch.log_softmax(logits, dim=-1).gather(-1, next_tokens.unsqueeze(-1)).squeeze(0).squeeze(-1)

        prompt_len = len(prompt_ids)
        scored = []
        for resp_idx in range(usable_response_len):
            absolute_pos = prompt_len + resp_idx
            pred_pos = absolute_pos - 1
            if pred_pos < 0 or pred_pos >= logps.numel():
                continue
            scored.append((float(logps[pred_pos].item()), resp_idx))
        if not scored:
            return uniform_boundaries(len(response_ids), count)
        scored.sort(key=lambda x: x[0])
        selected = sorted(idx for _, idx in scored[: min(count, len(scored))])
        if count > len(selected):
            selected.extend(uniform_boundaries(len(response_ids), count - len(selected)))
        return sorted(selected)


def apply_pause_method(sample: RawSample, tokenizer, pause_id: int, args, group_count: int, dit_scorer=None) -> dict:
    if getattr(args, "domain", "math") == "code":
        return apply_code_pause_method(sample, tokenizer, args, group_count, dit_scorer)
    group_ids = pause_ids(pause_id, args.pause_tokens_per_step)
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(sample.response, add_special_tokens=False)
    rng = random.Random(args.seed + int(sample.metadata["source_index"]))

    if args.method == "none":
        group_count = 0
    elif args.method == "append":
        prompt_ids = prompt_ids + group_ids * group_count
    elif args.method == "random":
        response_ids = insert_groups(response_ids, random_boundaries(len(response_ids), group_count, rng), group_ids)
    elif args.method == "dit":
        if dit_scorer is None:
            raise RuntimeError("DIT method requires a DitScorer.")
        boundaries = dit_scorer.lowest_logprob_boundaries(prompt_ids, response_ids, group_count)
        response_ids = insert_groups(response_ids, boundaries, group_ids)
    elif args.method == "step":
        if sample.steps:
            response_ids = []
            for step in sample.steps:
                response_ids.extend(group_ids)
                response_ids.extend(tokenizer.encode(step, add_special_tokens=False))
                response_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
        else:
            response_ids = group_ids + response_ids
    else:
        raise ValueError(f"Unsupported pause method: {args.method}")

    return {
        "question": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "answer": tokenizer.decode(response_ids, skip_special_tokens=False),
        "pause_metadata": {
            **sample.metadata,
            "step_count": sample.step_count,
            "pause_group_count": group_count,
            "pause_token_count": group_count * args.pause_tokens_per_step,
        },
    }


def apply_code_pause_method(sample, tokenizer, args, group_count, dit_scorer=None):
    prompt, response = sample.prompt, sample.response
    pause_text = args.pause_token * args.pause_tokens_per_step
    if args.method == "none":
        group_count = 0
    elif args.method == "append":
        prompt += pause_text * group_count
    elif args.method in ("step", "code_line"):
        response, group_count = insert_pauses_before_code_lines(response, pause_text)
    elif args.method in ("random", "dit"):
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if args.method == "random":
            rng = random.Random(args.seed + int(sample.metadata["source_index"]))
            boundaries = random_boundaries(len(response_ids), group_count, rng)
        else:
            if dit_scorer is None:
                raise RuntimeError("DIT method requires a DitScorer.")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            boundaries = dit_scorer.lowest_logprob_boundaries(prompt_ids, response_ids, group_count)
        response = insert_at_token_boundaries(response, boundaries, tokenizer, pause_text)
        group_count = len(boundaries)
    else:
        raise ValueError(f"Unsupported code pause method: {args.method}")
    return {
        "question": prompt,
        "answer": response,
        "pause_metadata": {
            **sample.metadata, "domain": "code", "step_count": sample.step_count,
            "pause_group_count": group_count,
            "pause_token_count": group_count * args.pause_tokens_per_step,
        },
    }


def save_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pause_dataset(args) -> None:
    domain = getattr(args, "domain", "math")
    if args.method == "code_line" and domain != "code":
        raise ValueError("--method code_line requires --domain code.")
    if domain == "code" and args.disable_fast_tokenizer:
        raise ValueError("Code datasets require a fast tokenizer to preserve source text at token boundaries.")
    if args.max_length < 3:
        raise ValueError("--max_length must be at least 3.")
    if not 0 <= args.eval_split_ratio < 1:
        raise ValueError("--eval_split_ratio must be in [0, 1).")
    dataset = load_source(args.dataset, split=args.dataset_split, config=args.dataset_config,
                          revision=args.dataset_revision, max_samples=args.max_samples)
    validate_columns(dataset, args.input_key, args.output_key)
    slug = dataset_slug(args.dataset)
    default_output_root = os.path.join(args.output_root, slug)
    tokenizer_save_path = args.pause_tokenizer_save_path or os.path.join(default_output_root, "pause_tokenizer")
    tokenizer, pause_id = ensure_pause_tokenizer(
        args.tokenizer_path,
        tokenizer_save_path,
        args.pause_token,
        use_fast=not args.disable_fast_tokenizer,
    )
    # Preserve the historical corpus tokenizers: Qwen stored an ordinary
    # single-word added token; Llama stored a special token without that rule.
    tokenizer.add_tokens(
        [AddedToken(args.pause_token, normalized=False,
                    single_word=args.pause_tokenizer_single_word,
                    special=args.pause_tokenizer_special)],
        special_tokens=args.pause_tokenizer_special,
    )
    if domain == "code":
        tokenizer.clean_up_tokenization_spaces = False
    tokenizer.save_pretrained(tokenizer_save_path)
    pause_id = one_token_id(tokenizer, args.pause_token)
    if pause_id is None or (args.expected_pause_token_id is not None and pause_id != args.expected_pause_token_id):
        raise ValueError(f"Unexpected pause token id: {pause_id}; expected {args.expected_pause_token_id}.")

    splitter_fn = None if domain == "code" else load_splitter(args.splitter)
    raw_samples = load_raw_samples(dataset, args, tokenizer, splitter_fn)
    if not raw_samples:
        raise ValueError("No samples remain after prompt-length filtering; shorten prompts or increase --max_length.")

    avg_step_count = sum(x.step_count for x in raw_samples) / len(raw_samples)
    avg_group_count = max(1, int(math.floor(avg_step_count + 0.5)))

    if args.save_path:
        save_path = args.save_path
    elif args.method == "none":
        save_path = os.path.join(default_output_root, "none")
    else:
        save_path = os.path.join(
            default_output_root,
            f"{args.method}_{args.budget_mode}_pps{args.pause_tokens_per_step}",
        )
    os.makedirs(save_path, exist_ok=True)

    dit_scorer = None
    if args.method == "dit":
        dit_scorer = DitScorer(args.dit_model_path, tokenizer, args.dit_max_length or args.max_length)

    processed = []
    for sample in tqdm(raw_samples, desc=f"Building {args.method}/{args.budget_mode}"):
        if args.method == "none":
            group_count = 0
        elif args.method == "step":
            group_count = sample.step_count
        elif args.budget_mode == "exact_steps":
            group_count = sample.step_count
        else:
            group_count = avg_group_count
        processed.append(apply_pause_method(sample, tokenizer, pause_id, args, group_count, dit_scorer=dit_scorer))

    rng = random.Random(args.seed)
    rng.shuffle(processed)
    eval_size = int(len(processed) * args.eval_split_ratio)
    eval_rows = processed[:eval_size] if eval_size > 0 else []
    train_rows = processed[eval_size:]

    train_path = os.path.join(save_path, "train_pause_dataset.jsonl")
    eval_path = os.path.join(save_path, "eval_pause_dataset.jsonl")
    save_jsonl(train_path, train_rows)
    save_jsonl(eval_path, eval_rows)

    metadata = {
        "domain": domain,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "max_samples": args.max_samples,
        "num_source_samples": len(dataset),
        "num_filtered_samples": len(dataset) - len(raw_samples),
        "dataset_slug": slug,
        "tokenizer_path": args.tokenizer_path,
        "pause_tokenizer_path": tokenizer_save_path,
        "pause_token": args.pause_token,
        "pause_token_id": pause_id,
        "input_key": args.input_key,
        "output_key": args.output_key,
        "save_input_key": "question",
        "save_output_key": "answer",
        "method": "step" if args.method == "code_line" else args.method,
        "budget_mode": args.budget_mode,
        "pause_tokens_per_step": args.pause_tokens_per_step,
        "avg_step_count": avg_step_count,
        "avg_group_count": 0 if args.method == "none" else avg_group_count,
        "avg_pause_token_count": 0 if args.method == "none" else avg_group_count * args.pause_tokens_per_step,
        "num_samples": len(processed),
        "num_train_samples": len(train_rows),
        "num_eval_samples": len(eval_rows),
        "train_path": train_path,
        "eval_path": eval_path if eval_rows else None,
        "split_mode": args.split_mode,
        "splitter": args.splitter,
        "seed": args.seed,
        "eval_split_ratio": args.eval_split_ratio,
        "dit_model_path": args.dit_model_path if args.method == "dit" else None,
        "pause_tokenizer_single_word": args.pause_tokenizer_single_word,
        "pause_tokenizer_special": args.pause_tokenizer_special,
    }
    if domain == "code":
        metadata["insertion_method"] = "code_line" if args.method == "step" else args.method
    with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved pause dataset to {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--domain", choices=("math", "code"), default="math")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="pause_dataset")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--pause_tokenizer_save_path", type=str, default=None)
    parser.add_argument("--pause_token", type=str, default=PAUSE_TOKEN)
    parser.add_argument("--pause_tokenizer_single_word", action="store_true")
    parser.add_argument("--pause_tokenizer_special", action="store_true")
    parser.add_argument("--expected_pause_token_id", type=int)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--budget_mode", type=str, required=True, choices=BUDGET_MODES)
    parser.add_argument("--pause_tokens_per_step", type=int, default=1)
    parser.add_argument("--dit_model_path", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dit_max_length", type=int, default=None)

    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--output_key", type=str, default="answer")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--dataset_revision", type=str)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split_mode", type=str, default="sentence", choices=["sentence", "chunk"])
    parser.add_argument("--splitter", type=str, default="atomic_math")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    build_pause_dataset(parser.parse_args())
