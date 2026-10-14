"""Small, dependency-light helpers shared by the reproduction commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a configuration mapping: {path}")
    return data


def cli_args(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + name)
        if value is True:
            continue
        if isinstance(value, (list, tuple)):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def gpu_count(explicit: int | None = None) -> int:
    if explicit is not None:
        count = explicit
    elif os.environ.get("NUM_GPUS"):
        count = int(os.environ["NUM_GPUS"])
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        count = len([device for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                     if device.strip() and device.strip() != "-1"])
    elif os.environ.get("SLURM_GPUS_ON_NODE", "").isdigit():
        count = int(os.environ["SLURM_GPUS_ON_NODE"])
    else:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            count = len(completed.stdout.strip().splitlines())
        except (OSError, subprocess.SubprocessError):
            # Permit CPU-side command inspection. The actual launcher reports
            # the unavailable accelerator when a training job is requested.
            count = 1
    if count < 1:
        raise ValueError("At least one GPU is required; use --num-gpus with a positive integer")
    return count


def torchrun(module: str, arguments: list[str], num_gpus: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", f"--nproc_per_node={num_gpus}", "--module", module, *arguments]


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[redacted]" if any(part in key.lower() for part in
                ("api_key", "password", "access_token", "secret")) else _scrub(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def launch(command: list[str], output_dir: Path | str | None,
           dry_run: bool = False, metadata: dict[str, Any] | None = None) -> None:
    command = [str(part) for part in command]
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    manifest = {"command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                "configuration": metadata or {}}
    manifest_path = None
    if output_dir is not None:
        directory = Path(output_dir)
        if not directory.is_absolute():
            directory = ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        # Preserve earlier invocations (e.g. two evaluation suites or resume).
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        manifest_path = directory / f"invocation_{stamp}.json"
        manifest_path.write_text(json.dumps(_scrub(manifest), indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(command, cwd=ROOT)
    if manifest_path is not None:
        manifest["exit_code"] = result.returncode
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(_scrub(manifest), indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)
