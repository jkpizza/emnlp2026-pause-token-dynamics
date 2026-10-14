#!/usr/bin/env python3
"""Run LightEval with vLLM data-parallel performance patches.

LightEval launches one Ray task per data-parallel replica. In the installed
LightEval version, each replica also forces vLLM to use Ray as its internal
distributed executor whenever data_parallel_size > 1. For single-GPU replicas
that nested Ray executor is unnecessary and can overload Ray startup on 4 GPU
nodes. Keep outer LightEval Ray data parallelism, but let each single-GPU vLLM
replica use vLLM's local executor.

LightEval also creates those Ray workers inside every _generate() call, which
reloads the model separately for GENERATIVE and LOGPROBS requests. This wrapper
keeps one Ray actor per data-parallel replica alive for the full LightEval model
lifetime so each actor initializes vLLM's LLM class once and reuses it.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any


def _enabled(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).lower() not in {"0", "false", "no", "off"}


class _PersistentVLLMWorker:
    def __init__(self, model_args: dict[str, Any]):
        from vllm import LLM

        self.llm = LLM(**model_args)

    def generate(self, requests: list[list[int]], sampling_params: Any) -> list[Any]:
        if not requests:
            return []
        return self.llm.generate(prompt_token_ids=requests, sampling_params=sampling_params)

    def close(self) -> bool:
        import gc

        try:
            engine = getattr(self.llm, "llm_engine", None)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass

        self.llm = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return True


def patch_lighteval_vllm_backend() -> None:
    import lighteval.models.vllm.vllm_model as vllm_model

    if getattr(vllm_model.VLLMModel, "_latent_steering_vllm_patch", False):
        return

    original_generate = vllm_model.VLLMModel._generate
    original_cleanup = vllm_model.VLLMModel.cleanup
    def patched_create_auto_model(self, config):
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enable_prefix_caching": config.enable_prefix_caching,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": _enabled(
                os.environ.get("LIGHTEVAL_VLLM_ENFORCE_EAGER"),
                default=True,
            ),
        }

        if config.quantization is not None:
            self.model_args["quantization"] = config.quantization
        if config.load_format is not None:
            self.model_args["load_format"] = config.load_format

        if config.data_parallel_size > 1:
            if config.tensor_parallel_size > 1 or config.pipeline_parallel_size > 1:
                self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"

            if self._max_length is None:
                vllm_model.logger.warning(
                    "The model max_length was not set in the model arguments. Since the model is using data parallelism, it is created later "
                    " with `ray`, so we can't infer the max_length automatically atm. It might raise issues later on: if it does, relaunch your "
                    "run, but set `max_model_length` explicitely in the model args."
                )
            return None

        model = vllm_model.LLM(**self.model_args)

        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def get_persistent_dp_actors(self):
        actors = getattr(self, "_latent_steering_dp_actors", None)
        if actors is not None:
            return actors

        remote_worker = vllm_model.ray.remote(_PersistentVLLMWorker).options(
            num_gpus=int(self.tensor_parallel_size)
        )
        actors = [
            remote_worker.remote(dict(self.model_args))
            for _ in range(int(self.data_parallel_size))
        ]
        self._latent_steering_dp_actors = actors
        return actors

    def patched_generate(
        self,
        inputs: list[list[int]],
        max_new_tokens=None,
        stop_tokens=None,
        returns_logits=False,
        num_samples=1,
        generate=True,
    ) -> list[Any]:
        if (
            int(getattr(self, "data_parallel_size", 1)) <= 1
            or not _enabled(os.environ.get("LIGHTEVAL_VLLM_PERSISTENT_DP"), default=True)
        ):
            return original_generate(
                self,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=returns_logits,
                num_samples=num_samples,
                generate=generate,
            )

        sampling_params = vllm_model.SamplingParams(
            **self.config.generation_parameters.to_vllm_dict()
        )

        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0
            if num_samples > 1 and sampling_params.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        requests = [list(x) for x in vllm_model.distribute(int(self.data_parallel_size), inputs)]
        actors = get_persistent_dp_actors(self)

        object_refs = [
            actor.generate.remote(request, sampling_params)
            for actor, request in zip(actors, requests)
        ]
        results = vllm_model.ray.get(object_refs)

        return [
            item
            for item in itertools.chain.from_iterable(
                itertools.zip_longest(*[list(result) for result in results])
            )
            if item is not None
        ]

    def patched_cleanup(self):
        actors = getattr(self, "_latent_steering_dp_actors", None)
        if actors:
            try:
                vllm_model.ray.get([actor.close.remote() for actor in actors], timeout=60)
            except Exception:
                pass
            for actor in actors:
                try:
                    vllm_model.ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            self._latent_steering_dp_actors = None

        return original_cleanup(self)

    vllm_model.VLLMModel._create_auto_model = patched_create_auto_model
    vllm_model.VLLMModel._generate = patched_generate
    vllm_model.VLLMModel.cleanup = patched_cleanup
    vllm_model.VLLMModel._latent_steering_vllm_patch = True


def main() -> None:
    if _enabled(os.environ.get("LIGHTEVAL_PATCH_DP_SINGLE_GPU_BACKEND")):
        patch_lighteval_vllm_backend()

    from lighteval.__main__ import app

    sys.argv[0] = "lighteval"
    app()


if __name__ == "__main__":
    main()
