#!/usr/bin/env python3
"""Compute AUC tables from W&B-exported CSV files."""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


IGNORED_SUFFIXES = ("__MIN", "__MAX")
STEP_COLUMNS = {"step", "global_step", "_step"}


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


def safe_slug(value: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.=-]+", "_", value).strip("_")
    return (slug or "value")[:max_len]


def split_export_column(column: str) -> tuple[str, str] | None:
    if " - " not in column:
        return None
    run_name, metric = column.rsplit(" - ", 1)
    metric = metric.strip()
    if metric.lower() in STEP_COLUMNS:
        return None
    if metric.endswith(IGNORED_SUFFIXES):
        return None
    return run_name.strip(), metric


def infer_method(run_name: str) -> str:
    lower = run_name.lower()
    if lower.startswith("pause_") or "_pause" in lower or "-pause" in lower:
        return "pause"
    if lower.startswith("normal_") or "_normal" in lower or "-normal" in lower:
        return "normal"
    return "unknown"


def infer_base_model(run_name: str, csv_path: Path) -> str:
    match = re.search(r"(Qwen\d+(?:[.-]\d+)?-[0-9.]+B-Base)", run_name, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    filename = csv_path.name.lower()
    if "1_7b" in filename or "1.7b" in filename:
        return "Qwen3-1.7B-Base"
    if "4b" in filename:
        return "Qwen3-4B-Base"
    return "unknown"


def infer_csv_label(csv_path: Path) -> str:
    name = csv_path.stem
    name = re.sub(r"^wandb_export_", "", name)
    return name


def prepare_curve(df: pd.DataFrame, step_column: str, value_column: str) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for row_order, row in df.iterrows():
        step = coerce_float(row.get(step_column))
        value = coerce_float(row.get(value_column))
        if step is None or value is None:
            continue
        records.append({"step": step, "value": value, "_row_order": int(row_order)})

    if not records:
        return pd.DataFrame(columns=["step", "value"])

    curve = pd.DataFrame.from_records(records)
    curve = curve.sort_values(["step", "_row_order"]).drop_duplicates(subset=["step"], keep="last")
    return curve.sort_values("step")[["step", "value"]].reset_index(drop=True)


def compute_auc(curve: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if len(curve) < 2:
        return None, None, None
    steps = curve["step"].to_numpy(dtype=float)
    values = curve["value"].to_numpy(dtype=float)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    raw_auc = float(integrate(values, steps))
    step_range = float(steps[-1] - steps[0])
    normalized_auc = None if step_range == 0.0 else raw_auc / step_range
    return raw_auc, normalized_auc, step_range


def analyze_csv(csv_path: Path, curves_dir: Path, write_curves: bool) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    step_column = df.columns[0]
    rows: list[dict[str, Any]] = []
    for column in df.columns[1:]:
        parsed = split_export_column(column)
        if parsed is None:
            continue
        run_name, metric = parsed
        curve = prepare_curve(df, step_column=step_column, value_column=column)
        raw_auc, normalized_auc, step_range = compute_auc(curve)
        status = "ok" if len(curve) >= 2 and step_range not in (None, 0.0) else "insufficient_points"

        curve_path = None
        if write_curves and not curve.empty:
            curve_path = curves_dir / f"{safe_slug(csv_path.stem)}__{safe_slug(run_name)}__{safe_slug(metric)}.csv"
            curve_out = curve.copy()
            curve_out.insert(0, "source_csv", csv_path.name)
            curve_out.insert(1, "run_name", run_name)
            curve_out.insert(2, "method", infer_method(run_name))
            curve_out.insert(3, "base_model", infer_base_model(run_name, csv_path))
            curve_out.insert(4, "metric", metric)
            curve_out.to_csv(curve_path, index=False)

        rows.append(
            {
                "source_csv": csv_path.name,
                "csv_label": infer_csv_label(csv_path),
                "base_model": infer_base_model(run_name, csv_path),
                "method": infer_method(run_name),
                "run_name": run_name,
                "metric": metric,
                "points": int(len(curve)),
                "min_step": None if curve.empty else float(curve["step"].iloc[0]),
                "max_step": None if curve.empty else float(curve["step"].iloc[-1]),
                "step_range": step_range,
                "raw_auc": raw_auc,
                "normalized_auc": normalized_auc,
                "status": status,
                "curve_csv": str(curve_path.relative_to(curves_dir.parent)) if curve_path else None,
            }
        )
    return rows


def build_wide_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    index_cols = ["source_csv", "csv_label", "base_model", "metric"]
    value_cols = ["normalized_auc", "raw_auc", "points", "min_step", "max_step", "status"]
    pivot = summary.pivot_table(index=index_cols, columns="method", values=value_cols, aggfunc="first")
    pivot.columns = [f"{method}_{value}" for value, method in pivot.columns]
    pivot = pivot.reset_index()

    ordered_cols = list(index_cols)
    for method in ("pause", "normal", "unknown"):
        for value in value_cols:
            col = f"{method}_{value}"
            if col in pivot.columns:
                ordered_cols.append(col)
    remaining = [col for col in pivot.columns if col not in ordered_cols]
    return pivot[ordered_cols + remaining].sort_values(index_cols).reset_index(drop=True)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--csv-dir", default=str(script_dir / "csv"), help="Directory containing W&B CSV exports.")
    parser.add_argument(
        "--out-dir",
        default=str(script_dir / "results"),
        help="Output root. A csv_auc_<timestamp> subdirectory is created here.",
    )
    parser.add_argument("--pattern", default="*.csv", help="Input CSV glob pattern.")
    parser.add_argument("--timestamp", default=None, help="Optional output timestamp suffix.")
    parser.add_argument("--write-curves", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_dir = Path(args.csv_dir).expanduser()
    if not csv_dir.exists():
        parser.error(f"--csv-dir does not exist: {csv_dir}")

    csv_paths = sorted(csv_dir.glob(args.pattern))
    if not csv_paths:
        parser.error(f"No CSV files matched {args.pattern!r} in {csv_dir}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() / f"csv_auc_{timestamp}"
    curves_dir = out_dir / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_curves:
        curves_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        rows.extend(analyze_csv(csv_path, curves_dir=curves_dir, write_curves=args.write_curves))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["base_model", "csv_label", "metric", "method"]).reset_index(drop=True)
    wide = build_wide_table(summary)

    summary.to_csv(out_dir / "auc_summary.csv", index=False)
    wide.to_csv(out_dir / "auc_table.csv", index=False)
    write_markdown_table(summary, out_dir / "auc_summary.md")
    write_markdown_table(wide, out_dir / "auc_table.md")

    print(f"Wrote CSV AUC summary to {out_dir}")
    print(f"Processed {len(csv_paths)} CSV file(s), {len(summary)} method curve(s).")
    return out_dir


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
