import gc
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


NONE_LIKE_VALUES = {"", "none", "null"}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}
TEMP_MERGED_MARKER = ".latent_steering_eval_merged_model.json"


@dataclass(frozen=True)
class EvalModelConfig:
    requested_model_path: str
    is_lora_adapter: bool
    vllm_model_path: str
    tokenizer_path: str
    stop_model_path: str
    cleanup_path: Optional[str] = None


def normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    if stripped.lower() in NONE_LIKE_VALUES:
        return None
    return stripped


def adapter_config_path(model_path: Optional[str]) -> Optional[str]:
    model_path = normalize_optional_path(model_path)
    if not model_path:
        return None
    path = os.path.join(model_path, "adapter_config.json")
    return path if os.path.isfile(path) else None


def read_adapter_config(model_path: str) -> dict:
    config_path = adapter_config_path(model_path)
    if config_path is None:
        raise ValueError(f"{model_path} does not contain adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_lora_adapter_checkpoint(model_path: Optional[str]) -> bool:
    return adapter_config_path(model_path) is not None


def adapter_base_model_path(model_path: str) -> Optional[str]:
    return normalize_optional_path(read_adapter_config(model_path).get("base_model_name_or_path"))


def has_tokenizer_files(model_path: Optional[str]) -> bool:
    model_path = normalize_optional_path(model_path)
    if not model_path or not os.path.isdir(model_path):
        return False
    return any(os.path.isfile(os.path.join(model_path, name)) for name in TOKENIZER_FILES)


def marker_path(merged_path: str) -> str:
    return os.path.join(merged_path, TEMP_MERGED_MARKER)


def is_owned_temp_merged_model(merged_path: Optional[str]) -> bool:
    merged_path = normalize_optional_path(merged_path)
    if not merged_path or not os.path.isdir(merged_path):
        return False
    return os.path.isfile(marker_path(merged_path))


def cleanup_temp_merged_model_path(merged_path: Optional[str], print_fn=print) -> None:
    merged_path = normalize_optional_path(merged_path)
    if not merged_path or not os.path.exists(merged_path):
        return
    if not is_owned_temp_merged_model(merged_path):
        raise RuntimeError(f"Refusing to delete unmarked merged model directory: {merged_path}")
    shutil.rmtree(merged_path)
    print_fn(f"Deleted temporary merged LoRA model: {merged_path}")


def cleanup_eval_model_config(config: EvalModelConfig, rank: int = 0, world_size: int = 1, print_fn=print) -> None:
    if not config.cleanup_path:
        return

    _barrier(world_size)
    if rank == 0:
        cleanup_temp_merged_model_path(config.cleanup_path, print_fn=print_fn)
    _barrier(world_size)


def release_model_for_cleanup(*objects) -> None:
    for obj in objects:
        if obj is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception:
                    pass
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_eval_model_config(
    pretrain: str,
    lora_base_model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    print_fn=print,
    merged_path: Optional[str] = None,
) -> EvalModelConfig:
    pretrain = normalize_optional_path(pretrain)
    if pretrain is None:
        raise ValueError("--pretrain must be provided")

    tokenizer_path = normalize_optional_path(tokenizer_path)
    if not is_lora_adapter_checkpoint(pretrain):
        return EvalModelConfig(
            requested_model_path=pretrain,
            is_lora_adapter=False,
            vllm_model_path=pretrain,
            tokenizer_path=tokenizer_path or pretrain,
            stop_model_path=pretrain,
        )

    base_model_path = normalize_optional_path(lora_base_model_path) or adapter_base_model_path(pretrain)
    if base_model_path is None:
        raise ValueError(
            f"{pretrain} is a LoRA adapter checkpoint because it contains adapter_config.json, "
            "but no base model was provided via --lora_base_model_path and adapter_config.json "
            "does not contain base_model_name_or_path."
        )

    merged_path = merged_path or pretrain + "_merged"
    tokenizer_source = tokenizer_path or (pretrain if has_tokenizer_files(pretrain) else base_model_path)
    result = {"ok": True, "error": None}

    if rank == 0:
        try:
            merge_lora_adapter_checkpoint(
                adapter_path=pretrain,
                base_model_path=base_model_path,
                merged_path=merged_path,
                tokenizer_source=tokenizer_source,
                print_fn=print_fn,
            )
        except Exception:
            result = {"ok": False, "error": traceback.format_exc()}

    result = _broadcast_rank0_result(result, world_size)
    if not result["ok"]:
        raise RuntimeError(f"Failed to merge LoRA adapter checkpoint {pretrain}:\n{result['error']}")

    return EvalModelConfig(
        requested_model_path=pretrain,
        is_lora_adapter=True,
        vllm_model_path=merged_path,
        tokenizer_path=merged_path,
        stop_model_path=base_model_path,
        cleanup_path=merged_path,
    )


def merge_lora_adapter_checkpoint(
    adapter_path: str,
    base_model_path: str,
    merged_path: str,
    tokenizer_source: str,
    print_fn=print,
) -> None:
    if os.path.exists(merged_path):
        if is_owned_temp_merged_model(merged_path):
            cleanup_temp_merged_model_path(merged_path, print_fn=print_fn)
        else:
            raise FileExistsError(
                f"Refusing to overwrite existing unmarked merged model directory: {merged_path}"
            )

    os.makedirs(merged_path, exist_ok=False)
    _write_marker(merged_path, adapter_path, base_model_path)

    try:
        import torch
        from peft import PeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print_fn(f"Merging LoRA adapter for evaluation: adapter={adapter_path}, base={base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _maybe_resize_embeddings(model, tokenizer, print_fn=print_fn)
        model_to_merge = PeftModelForCausalLM.from_pretrained(model, adapter_path)
        merged_model = model_to_merge.merge_and_unload()

        tokenizer.save_pretrained(merged_path)
        merged_model.save_pretrained(save_directory=merged_path)
        output_config_file = os.path.join(merged_path, "config.json")
        merged_model.config.to_json_file(output_config_file)
        _write_marker(merged_path, adapter_path, base_model_path)
        print_fn(f"Saved temporary merged LoRA model to {merged_path}")

        del merged_model
        del model_to_merge
        del model
        del tokenizer
        release_model_for_cleanup()
    except Exception:
        if is_owned_temp_merged_model(merged_path):
            shutil.rmtree(merged_path, ignore_errors=True)
        raise


def _maybe_resize_embeddings(model, tokenizer, print_fn=print) -> None:
    try:
        embedding_count = model.get_input_embeddings().num_embeddings
        tokenizer_count = len(tokenizer)
    except Exception:
        return
    if tokenizer_count > embedding_count:
        print_fn(f"Resizing base model embeddings from {embedding_count} to {tokenizer_count}")
        model.resize_token_embeddings(tokenizer_count)


def _write_marker(merged_path: str, adapter_path: str, base_model_path: str) -> None:
    marker = {
        "created_by": "latent_steering_evaluation_lora_merge",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "adapter_path": adapter_path,
        "base_model_path": base_model_path,
    }
    with open(marker_path(merged_path), "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)


def _broadcast_rank0_result(result: dict, world_size: int) -> dict:
    if world_size <= 1:
        return result

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return result
    objects = [result if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def _barrier(world_size: int) -> None:
    if world_size <= 1:
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
