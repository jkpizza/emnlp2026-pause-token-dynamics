#!/usr/bin/env python3
"""Fetch W&B eval curves and compute AUC metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "eval/math_500/pass@4",
    "eval/math_500/pass@1",
    "eval/gsm8k/pass@4",
    "eval/gsm8k/pass@1",
)


@dataclass
class MetricFetchResult:
    requested_metric: str
    resolved_metric: str | None
    step_key: str | None
    rows_scanned: int
    curve: pd.DataFrame
    status: str
    message: str = ""


def parse_list_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        for item in re.split(r"[,\n]", value):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def read_runs_file(path: str | Path) -> list[str]:
    runs: list[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                runs.append(line)
    return runs


def metric_aliases(metric: str) -> list[str]:
    aliases = [metric]
    if "pass@" in metric:
        aliases.append(metric.replace("pass@", "pass_at_"))
        aliases.append(metric.replace("pass@", "pass"))
    elif "pass_at_" in metric:
        aliases.append(metric.replace("pass_at_", "pass@"))
        aliases.append(metric.replace("pass_at_", "pass"))
    return dedupe_preserve_order(aliases)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def safe_slug(value: str, max_len: int = 160) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.=-]+", "_", value).strip("_")
    return (slug or "value")[:max_len]


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def prepare_curve(
    rows: Sequence[dict[str, Any]],
    metric_key: str,
    step_key: str,
    min_step: float | None = None,
    max_step: float | None = None,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for row_order, row in enumerate(rows):
        step = coerce_float(row.get(step_key))
        value = coerce_float(row.get(metric_key))
        if step is None or value is None:
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        records.append({"step": step, "value": value, "_row_order": row_order})

    if not records:
        return pd.DataFrame(columns=["step", "value"])

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["step", "_row_order"]).drop_duplicates(subset=["step"], keep="last")
    df = df.sort_values("step")[["step", "value"]].reset_index(drop=True)
    return df


def compute_auc(steps: Sequence[float], scores: Sequence[float]) -> tuple[float, float | None, float]:
    steps_array = np.asarray(steps, dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    raw_auc = float(integrate(scores_array, steps_array))
    step_range = float(steps_array[-1] - steps_array[0])
    normalized_auc = None if step_range == 0.0 else raw_auc / step_range
    return raw_auc, normalized_auc, step_range


def history_key_sets(metric_key: str, step_key: str) -> list[tuple[str, list[str]]]:
    if step_key == "global_step":
        return [("global_step", [metric_key, "global_step"])]
    if step_key == "_step":
        return [("_step", [metric_key])]
    return [
        ("global_step", [metric_key, "global_step"]),
        ("_step", [metric_key]),
    ]


def fetch_metric_curve(
    run: Any,
    requested_metric: str,
    step_key: str,
    page_size: int,
    min_step: float | None = None,
    max_step: float | None = None,
) -> MetricFetchResult:
    errors: list[str] = []
    for metric_key in metric_aliases(requested_metric):
        for resolved_step_key, keys in history_key_sets(metric_key, step_key):
            try:
                rows = list(run.scan_history(keys=keys, page_size=page_size))
            except Exception as exc:  # pragma: no cover - exercised against live W&B
                errors.append(f"{metric_key} via {resolved_step_key}: {exc}")
                continue

            curve = prepare_curve(
                rows,
                metric_key=metric_key,
                step_key=resolved_step_key,
                min_step=min_step,
                max_step=max_step,
            )
            if not curve.empty:
                return MetricFetchResult(
                    requested_metric=requested_metric,
                    resolved_metric=metric_key,
                    step_key=resolved_step_key,
                    rows_scanned=len(rows),
                    curve=curve,
                    status="ok",
                )

    message = "; ".join(errors)
    status = "fetch_error" if errors else "no_data"
    return MetricFetchResult(
        requested_metric=requested_metric,
        resolved_metric=None,
        step_key=None,
        rows_scanned=0,
        curve=pd.DataFrame(columns=["step", "value"]),
        status=status,
        message=message,
    )


def run_metadata(run: Any, run_path: str) -> dict[str, Any]:
    return {
        "run_path": run_path,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "run_state": getattr(run, "state", None),
        "run_url": getattr(run, "url", None),
    }


def write_curve(curve: pd.DataFrame, path: Path, metadata: dict[str, Any], result: MetricFetchResult) -> None:
    out = curve.copy()
    out.insert(0, "run_path", metadata["run_path"])
    out.insert(1, "run_name", metadata.get("run_name"))
    out.insert(2, "requested_metric", result.requested_metric)
    out.insert(3, "resolved_metric", result.resolved_metric)
    out.insert(4, "step_key", result.step_key)
    out.to_csv(path, index=False)


def write_plot(curve: pd.DataFrame, path: Path, metadata: dict[str, Any], result: MetricFetchResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["step"], curve["value"], marker="o", linewidth=1.5, markersize=3)
    ax.set_xlabel(result.step_key or "step")
    ax.set_ylabel(result.requested_metric)
    title_name = metadata.get("run_name") or metadata["run_path"]
    ax.set_title(f"{title_name}\n{result.resolved_metric or result.requested_metric}", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary_row(
    metadata: dict[str, Any],
    result: MetricFetchResult,
    curve_csv: Path | None,
    plot_png: Path | None,
    normalize: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **metadata,
        "requested_metric": result.requested_metric,
        "resolved_metric": result.resolved_metric,
        "step_key": result.step_key,
        "status": result.status,
        "message": result.message,
        "points": int(len(result.curve)),
        "rows_scanned": int(result.rows_scanned),
        "min_step": None,
        "max_step": None,
        "step_range": None,
        "raw_auc": None,
        "normalized_auc": None,
        "auc": None,
        "auc_type": "normalized" if normalize else "raw",
        "curve_csv": str(curve_csv) if curve_csv else None,
        "plot_png": str(plot_png) if plot_png else None,
    }

    if result.status != "ok":
        return row

    if len(result.curve) < 2:
        row["status"] = "insufficient_points"
        row["message"] = "Need at least two distinct finite step/value points."
        return row

    steps = result.curve["step"].to_numpy(dtype=float)
    values = result.curve["value"].to_numpy(dtype=float)
    raw_auc, normalized_auc, step_range = compute_auc(steps, values)
    row.update(
        {
            "min_step": float(steps[0]),
            "max_step": float(steps[-1]),
            "step_range": float(step_range),
            "raw_auc": raw_auc,
            "normalized_auc": normalized_auc,
            "auc": normalized_auc if normalize else raw_auc,
        }
    )
    if normalized_auc is None:
        row["status"] = "zero_step_range"
        row["message"] = "Step range must be non-zero for normalized AUC."
        row["auc"] = raw_auc if not normalize else None
    return row


def analyze_runs(args: argparse.Namespace, api: Any) -> Path:
    out_root = Path(args.out_dir).expanduser()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = out_root / timestamp
    curves_dir = run_out_dir / "curves"
    plots_dir = run_out_dir / "plots"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_list_values(args.metrics) or list(DEFAULT_METRICS)
    run_paths = parse_list_values(args.runs)
    if args.runs_file:
        run_paths.extend(read_runs_file(args.runs_file))
    run_paths = dedupe_preserve_order(run_paths)
    if not run_paths:
        raise ValueError("At least one run must be provided with --runs or --runs-file.")

    summary_rows: list[dict[str, Any]] = []
    for run_path in run_paths:
        try:
            run = api.run(run_path)
        except Exception as exc:  # pragma: no cover - exercised against live W&B
            for metric in metrics:
                summary_rows.append(
                    {
                        "run_path": run_path,
                        "run_id": None,
                        "run_name": None,
                        "run_state": None,
                        "run_url": None,
                        "requested_metric": metric,
                        "resolved_metric": None,
                        "step_key": None,
                        "status": "run_error",
                        "message": str(exc),
                        "points": 0,
                        "rows_scanned": 0,
                        "min_step": None,
                        "max_step": None,
                        "step_range": None,
                        "raw_auc": None,
                        "normalized_auc": None,
                        "auc": None,
                        "auc_type": "normalized" if args.normalize else "raw",
                        "curve_csv": None,
                        "plot_png": None,
                    }
                )
            continue

        metadata = run_metadata(run, run_path)
        for metric in metrics:
            result = fetch_metric_curve(
                run,
                requested_metric=metric,
                step_key=args.step_key,
                page_size=args.page_size,
                min_step=args.min_step,
                max_step=args.max_step,
            )
            slug = f"{safe_slug(run_path)}__{safe_slug(metric)}"
            curve_csv = curves_dir / f"{slug}.csv" if not result.curve.empty else None
            plot_png = plots_dir / f"{slug}.png" if not result.curve.empty else None
            if curve_csv:
                write_curve(result.curve, curve_csv, metadata, result)
            if plot_png:
                write_plot(result.curve, plot_png, metadata, result)
            summary_rows.append(summary_row(metadata, result, curve_csv, plot_png, args.normalize))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_out_dir / "auc_summary.csv"
    summary_json = run_out_dir / "auc_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None, help="W&B run paths: ENTITY/PROJECT/RUN_ID")
    parser.add_argument("--runs-file", default=None, help="File with one W&B run path per line.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metric keys. Comma-separated values are accepted.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output root. A timestamped subdirectory is created inside this path.",
    )
    parser.add_argument("--page-size", type=int, default=5000, help="W&B scan_history page size.")
    parser.add_argument("--step-key", choices=("auto", "global_step", "_step"), default="auto")
    parser.add_argument("--min-step", type=float, default=None)
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp", default=None, help="Optional output subdirectory name.")
    return parser


def main(argv: Sequence[str] | None = None, api_factory: Callable[[], Any] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error("--page-size must be positive.")
    if args.min_step is not None and args.max_step is not None and args.min_step > args.max_step:
        parser.error("--min-step must be <= --max-step.")

    if api_factory is None:
        import wandb

        api_factory = wandb.Api

    out_dir = analyze_runs(args, api_factory())
    print(f"Wrote W&B AUC analysis results to {out_dir}")
    return out_dir


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
