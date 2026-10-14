import inspect
import json
import os
import re
from typing import Dict, List, Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


PAUSE_TOKEN = "<|PAUSE|>"


def single_token_id(tokenizer, token: str) -> Optional[int]:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if isinstance(ids, list) and len(ids) == 1:
        return int(ids[0])
    return None


def rewrite_tokenizer_files_with_pause(save_path: str, pause_token: str = PAUSE_TOKEN, print_fn=None) -> None:
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
        raise RuntimeError(
            "Pause tokenizer build expects a BPE-style tokenizer with tokenizer.json model.vocab or vocab.json."
        )

    if pause_token in vocab:
        return

    old_token = _pick_invalid_tokens(vocab, 1)[0]
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
            merged = "".join(line.split())
            if old_token not in merged:
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

    if print_fn is not None:
        print_fn(f"Pause tokenizer token replaced: {old_token!r} -> {pause_token!r} at id {token_id}")


def ensure_pause_tokenizer(
    pretrain_or_path: str,
    save_path: str,
    pause_token: str = PAUSE_TOKEN,
    use_fast: bool = True,
    padding_side: str = "left",
    print_fn=None,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Rank 0 owns the shared tokenizer path; all ranks reload it after the barrier.
    if rank == 0:
        needs_write = True
        if os.path.exists(save_path):
            try:
                existing = AutoTokenizer.from_pretrained(
                    save_path,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=use_fast,
                )
                needs_write = single_token_id(existing, pause_token) is None
            except Exception:
                pass

        if needs_write:
            tokenizer = AutoTokenizer.from_pretrained(pretrain_or_path, trust_remote_code=True, use_fast=use_fast)
            tokenizer.padding_side = padding_side
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save_pretrained(save_path)
            if single_token_id(tokenizer, pause_token) is None:
                rewrite_tokenizer_files_with_pause(save_path, pause_token, print_fn=print_fn)

    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        save_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=use_fast,
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ids = tokenizer.encode(pause_token, add_special_tokens=False)
    if not isinstance(ids, list) or len(ids) != 1:
        raise ValueError(f"Failed to create one-token {pause_token!r}; encoded ids={ids}")
    return tokenizer


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


def is_valid_generated_text(text: str) -> bool:
    return bool(VALID_LATEXY.match(text))


def _pick_invalid_tokens(vocab: Dict[str, int], need: int) -> List[str]:
    items = [(tok, tid, len(tok)) for tok, tid in vocab.items()]
    items.sort(key=lambda x: (-x[2], x[1]))
    out = []
    for tok, _, _ in items:
        if not is_valid_generated_text(tok):
            out.append(tok)
            if len(out) >= need:
                return out
    raise ValueError(f"Not enough invalid tokenizer tokens to replace; need {need}.")


class PauseLM(nn.Module):
    """Causal LM wrapper with one pause token and OpenRLHF Actor-style forward."""

    @staticmethod
    def _output_embeddings_tied(emb: nn.Module, out: Optional[nn.Module]) -> bool:
        if out is None:
            return False
        if out is emb:
            return True
        if not hasattr(out, "weight") or not hasattr(emb, "weight"):
            return False
        return out.weight.data_ptr() == emb.weight.data_ptr()

    @staticmethod
    def _single_token_id(tokenizer, token: str) -> Optional[int]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if isinstance(ids, list) and len(ids) == 1:
            return int(ids[0])
        return None

    @staticmethod
    def _mean_excluding(weight: torch.Tensor, excluded_ids: List[int]) -> torch.Tensor:
        mask = torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        valid_ids = [int(x) for x in excluded_ids if x is not None and 0 <= int(x) < weight.size(0)]
        if valid_ids:
            mask[torch.tensor(valid_ids, dtype=torch.long, device=weight.device)] = False
        if bool(mask.any().item()):
            value = weight.detach()[mask].float().mean(dim=0)
        else:
            value = weight.detach().float().mean(dim=0)
        return value.to(device=weight.device, dtype=weight.dtype)

    @staticmethod
    def _norm_mean_excluding(weight: torch.Tensor, excluded_ids: List[int]) -> torch.Tensor:
        mask = torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        valid_ids = [int(x) for x in excluded_ids if x is not None and 0 <= int(x) < weight.size(0)]
        if valid_ids:
            mask[torch.tensor(valid_ids, dtype=torch.long, device=weight.device)] = False
        source = weight.detach()[mask] if bool(mask.any().item()) else weight.detach()
        return source.float().norm(dim=-1).mean().clamp(min=1e-6)

    def _print(self, message: str) -> None:
        if hasattr(self, "strategy") and self.strategy is not None and hasattr(self.strategy, "print"):
            self.strategy.print(message)
        else:
            print(message)

    def _validate_pause_token(self) -> int:
        pause_id = self._single_token_id(self.tokenizer, self.pause_token)
        if pause_id is None:
            ids = self.tokenizer.encode(self.pause_token, add_special_tokens=False)
            raise ValueError(f"{self.pause_token!r} must tokenize to exactly one id, got {ids}")
        return pause_id

    def _save_existing_pause_tokenizer(self, save_path: str) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            os.makedirs(save_path, exist_ok=True)
            self.tokenizer.save_pretrained(save_path)
        if dist.is_initialized():
            dist.barrier()

    def _ensure_pause_tokenizer(self, save_path: str, use_fast: bool):
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Rank 0 owns the shared tokenizer path; all ranks reload it after the barrier.
        if rank == 0:
            needs_write = True
            if os.path.exists(save_path):
                try:
                    existing = AutoTokenizer.from_pretrained(
                        save_path,
                        local_files_only=True,
                        trust_remote_code=False,
                        use_fast=use_fast,
                    )
                    needs_write = self._single_token_id(existing, self.pause_token) is None
                except Exception:
                    pass

            if needs_write:
                os.makedirs(save_path, exist_ok=True)
                self.tokenizer.save_pretrained(save_path)
                if self._single_token_id(self.tokenizer, self.pause_token) is None:
                    self._rewrite_tokenizer_files_with_pause(save_path)

        if dist.is_initialized():
            dist.barrier()

        tokenizer = AutoTokenizer.from_pretrained(
            save_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=use_fast,
        )
        ids = tokenizer.encode(self.pause_token, add_special_tokens=False)
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError(f"Failed to create one-token {self.pause_token!r}; encoded ids={ids}")
        return tokenizer

    def _rewrite_tokenizer_files_with_pause(self, save_path: str) -> None:
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
            raise RuntimeError(
                "Pause tokenizer build expects a BPE-style tokenizer with tokenizer.json model.vocab or vocab.json."
            )

        if self.pause_token in vocab:
            return

        old_token = _pick_invalid_tokens(vocab, 1)[0]
        token_id = int(vocab.pop(old_token))
        vocab[self.pause_token] = token_id

        if isinstance(vocab_json_data, dict) and old_token in vocab_json_data:
            token_id_json = int(vocab_json_data.pop(old_token))
            vocab_json_data[self.pause_token] = token_id_json

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
                merged = "".join(line.split())
                if old_token not in merged:
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
            "content": self.pause_token,
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
                "content": self.pause_token,
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
            added_tokens[self.pause_token] = token_id
            with open(addedtoks, "w", encoding="utf-8") as f:
                json.dump(added_tokens, f, ensure_ascii=False, indent=2)

        self._print(f"Pause tokenizer token replaced: {old_token!r} -> {self.pause_token!r} at id {token_id}")

    def _initialize_pause_embedding(self, init_method: str) -> None:
        init_method = "none" if init_method is None else str(init_method)
        if init_method not in {"mean", "gaussian_norm", "none"}:
            raise ValueError("--pause_init_method must be one of: mean, gaussian_norm, none")

        pause_id = self._validate_pause_token()
        emb = self.model.get_input_embeddings()
        out = self.model.get_output_embeddings()

        if pause_id >= emb.weight.size(0):
            self._print(
                f"Pause token id {pause_id} exceeds input embedding size {emb.weight.size(0)}; resizing embeddings."
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            emb = self.model.get_input_embeddings()
            out = self.model.get_output_embeddings()

        out_tied = self._output_embeddings_tied(emb, out)
        with torch.no_grad():
            if init_method == "mean":
                emb_vec = self._mean_excluding(emb.weight, [pause_id])
                emb.weight.data[pause_id, :] = emb_vec.to(device=emb.weight.device, dtype=emb.weight.dtype)
                if out is not None and not out_tied:
                    out_vec = self._mean_excluding(out.weight, [pause_id])
                    out.weight.data[pause_id, :] = out_vec.to(device=out.weight.device, dtype=out.weight.dtype)
            elif init_method == "gaussian_norm":
                emb_norm = self._norm_mean_excluding(emb.weight, [pause_id]).to(emb.weight.device)
                vec = torch.randn(emb.weight.size(1), device=emb.weight.device, dtype=torch.float32)
                vec = vec / vec.norm().clamp(min=1e-6) * emb_norm
                emb.weight.data[pause_id, :] = vec.to(dtype=emb.weight.dtype)
                if out is not None and not out_tied:
                    out_norm = self._norm_mean_excluding(out.weight, [pause_id]).to(out.weight.device)
                    out_vec = torch.randn(out.weight.size(1), device=out.weight.device, dtype=torch.float32)
                    out_vec = out_vec / out_vec.norm().clamp(min=1e-6) * out_norm
                    out.weight.data[pause_id, :] = out_vec.to(dtype=out.weight.dtype)
            elif init_method == "none":
                pass

        try:
            self.model.tie_weights()
        except Exception:
            pass

        self.pause_token_id = pause_id
        self._print(f"Pause token id: {pause_id}; pause init method: {init_method}; output head tied={out_tied}")

    @staticmethod
    def _lora_config_supports(field_name: str) -> bool:
        try:
            return field_name in inspect.signature(LoraConfig).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_target_modules(target_modules):
        if isinstance(target_modules, tuple):
            target_modules = list(target_modules)
        if isinstance(target_modules, list) and len(target_modules) == 1 and target_modules[0] == "all-linear":
            return "all-linear"
        return target_modules

    @staticmethod
    def _is_all_linear_target(target_modules) -> bool:
        if target_modules == "all-linear":
            return True
        return isinstance(target_modules, (list, tuple)) and len(target_modules) == 1 and target_modules[0] == "all-linear"

    @staticmethod
    def _is_linear_like_module(module: nn.Module) -> bool:
        if isinstance(module, nn.Linear):
            return True
        if not hasattr(module, "weight"):
            return False
        class_name = module.__class__.__name__.lower()
        return class_name == "conv1d" or class_name.startswith("linear") or class_name.endswith("linear")

    @staticmethod
    def _append_unique(values: List[str], value: Optional[str]) -> List[str]:
        if value and value not in values:
            values.append(value)
        return values

    def _get_input_embedding_module_name(self) -> Optional[str]:
        emb = self.model.get_input_embeddings()
        if emb is None:
            return None
        for name, module in self.model.named_modules():
            if module is emb:
                return name
        return None

    def _expand_all_linear_target_modules(self) -> List[str]:
        input_emb = self.model.get_input_embeddings()
        output_emb = self.model.get_output_embeddings()
        target_names = []
        for name, module in self.model.named_modules():
            if not name or module is input_emb or module is output_emb:
                continue
            if self._is_linear_like_module(module):
                target_names.append(name)
        return target_names

    def _resolve_lora_embedding_config(self, target_modules, requested_mode: str):
        requested_mode = "embedding_lora" if requested_mode is None else str(requested_mode)
        valid_modes = {"embedding_lora", "pause_token", "full", "auto", "off"}
        if requested_mode not in valid_modes:
            raise ValueError(f"--lora_embedding_train_mode must be one of: {sorted(valid_modes)}")

        target_modules = self._normalize_target_modules(target_modules)
        embedding_name = self._get_input_embedding_module_name()

        def full_embedding_config(reason: Optional[str] = None):
            if embedding_name is None:
                raise ValueError("Cannot train PauseLM input embeddings because get_input_embeddings() was not found.")
            if not self._lora_config_supports("modules_to_save"):
                raise ValueError("Installed PEFT LoraConfig does not support modules_to_save.")
            if reason:
                self._print(reason)
            return target_modules, {"modules_to_save": [embedding_name]}, "full", embedding_name

        def pause_token_config(required: bool):
            if self._lora_config_supports("trainable_token_indices"):
                return (
                    target_modules,
                    {"trainable_token_indices": [int(self.pause_token_id)]},
                    "pause_token",
                    embedding_name,
                )
            if required:
                raise ValueError("Installed PEFT LoraConfig does not support trainable_token_indices.")
            return None

        def embedding_lora_config(required: bool):
            if embedding_name is None:
                if required:
                    return full_embedding_config(
                        "PauseLM embedding LoRA requested, but input embedding module name could not be resolved; "
                        "falling back to full embedding training."
                    )
                return None

            if self._is_all_linear_target(target_modules):
                expanded_targets = self._expand_all_linear_target_modules()
                if not expanded_targets:
                    if required:
                        return full_embedding_config(
                            "PauseLM embedding LoRA requested, but all-linear target expansion found no linear "
                            "modules; falling back to full embedding training."
                        )
                    return None
                expanded_targets = self._append_unique(expanded_targets, embedding_name)
                return expanded_targets, {}, "embedding_lora", embedding_name

            if isinstance(target_modules, (list, tuple)):
                expanded_targets = []
                for module_name in target_modules:
                    if module_name == "all-linear":
                        expanded_targets.extend(
                            name for name in self._expand_all_linear_target_modules() if name not in expanded_targets
                        )
                    elif module_name not in expanded_targets:
                        expanded_targets.append(module_name)
                expanded_targets = self._append_unique(expanded_targets, embedding_name)
                return expanded_targets, {}, "embedding_lora", embedding_name

            if required:
                return full_embedding_config(
                    "PauseLM embedding LoRA requested, but target_modules is not an explicit module list or "
                    "all-linear; falling back to full embedding training."
                )
            return None

        if requested_mode == "off":
            return target_modules, {}, "off", embedding_name
        if requested_mode == "pause_token":
            return pause_token_config(required=True)
        if requested_mode == "embedding_lora":
            return embedding_lora_config(required=True)
        if requested_mode == "full":
            return full_embedding_config()

        resolved = pause_token_config(required=False)
        if resolved is not None:
            return resolved
        resolved = embedding_lora_config(required=False)
        if resolved is not None:
            return resolved
        return full_embedding_config(
            "PauseLM auto LoRA embedding mode could not use pause-token or embedding LoRA; "
            "falling back to full embedding training."
        )

    def _log_trainable_parameters(
        self,
        lora_rank: int,
        requested_mode: str,
        resolved_mode: str,
        embedding_name: Optional[str],
    ):
        trainable = 0
        total = 0
        embedding_trainable = False
        for name, param in self.model.named_parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
                if embedding_name and embedding_name in name:
                    embedding_trainable = True

        ratio = 100.0 * trainable / max(total, 1)
        self._print(
            "PauseLM trainable params: "
            f"{trainable:,} || all params: {total:,} || trainable%: {ratio:.4f} || "
            f"lora_rank={lora_rank} || "
            f"requested_lora_embedding_train_mode={requested_mode} || "
            f"resolved_lora_embedding_train_mode={resolved_mode} || "
            f"pause_token_id={getattr(self, 'pause_token_id', None)} || "
            f"input_embedding_module={embedding_name} || "
            f"input_embedding_trainable={embedding_trainable}"
        )

    def __init__(
        self,
        pretrain_or_model,
        strategy,
        tokenizer=None,
        padding_side="right",
        pause_token: str = PAUSE_TOKEN,
        pause_init_method: str = "mean",
        pause_tokenizer_save_path: Optional[str] = None,
        use_fast=True,
        attn_implementation="flash_attention_2",
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        lora_embedding_train_mode="embedding_lora",
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.strategy = strategy
        self.pause_token = pause_token
        self.bf16 = bf16
        self._dschf = None

        if isinstance(pretrain_or_model, str):
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                self._dschf = HfDeepSpeedConfig(ds_config)

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    use_fast=use_fast,
                )
            self.tokenizer.padding_side = padding_side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                self.model.config.output_router_logits = True
                for module in self.model.modules():
                    if "SparseMoeBlock" in module.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [module.__class__])
                        self._print(f"Setting zero3 leaf for model class: {module.__class__.__name__}")
                        break

            self.model.config.use_cache = False
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model
            self.tokenizer = tokenizer
            self.packing_samples = packing_samples

        if self.tokenizer is None:
            raise ValueError("PauseLM requires a tokenizer.")

        tokenizer_save_path = pause_tokenizer_save_path or getattr(strategy.args, "save_path", "./pause_tokenizer")
        self.tokenizer = self._ensure_pause_tokenizer(tokenizer_save_path, use_fast=use_fast)
        self.tokenizer.padding_side = padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.model is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self._initialize_pause_embedding(pause_init_method)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pause_token]})
        resolved_lora_embedding_train_mode = "off"
        input_embedding_module_name = self._get_input_embedding_module_name()
        if lora_rank > 0:
            self.model.enable_input_require_grads()
            (
                target_modules,
                lora_embedding_kwargs,
                resolved_lora_embedding_train_mode,
                input_embedding_module_name,
            ) = self._resolve_lora_embedding_config(target_modules, lora_embedding_train_mode)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                **lora_embedding_kwargs,
            )
            self.model = get_peft_model(self.model, lora_config)

            if load_in_4bit:
                for name, module in self.model.named_modules():
                    if isinstance(module, LoraLayer):
                        module = module.to(torch.bfloat16)
                    if "norm" in name:
                        module = module.to(torch.float32)
                    if "lm_head" in name or "embed_tokens" in name:
                        if hasattr(module, "weight"):
                            module = module.to(torch.bfloat16)

        self._log_trainable_parameters(
            lora_rank,
            lora_embedding_train_mode,
            resolved_lora_embedding_train_mode,
            input_embedding_module_name,
        )

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    @property
    def config(self):
        return self.model.config

    def save_pretrained(self, *args, **kwargs):
        if isinstance(self.model, PeftModel):
            # PAUSE initialization changes frozen base rows before LoRA wraps
            # the model. Adapter deltas alone cannot reconstruct those rows.
            kwargs["save_embedding_layers"] = True
        return self.model.save_pretrained(*args, **kwargs)

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        batch, seqlen = sequences.size()
        forward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            forward_attention_mask = None
        else:
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=forward_attention_mask, position_ids=position_ids)
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
