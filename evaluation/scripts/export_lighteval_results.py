#!/usr/bin/env python3
"""Append flattened LightEval JSON metrics to CSV and TXT summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "run_timestamp",
    "model_name",
    "model_input",
    "tasks",
    "result_json",
    "lighteval_sha",
    "model_dtype",
    "model_size",
    "max_samples",
    "total_evaluation_time_seconds",
    "exported_at_utc",
]

TXT_PRIMARY_BENCHMARKS = ["all", "gpqa:mc", "mmlu_mc:_average", "hellaswag", "bigbench_hard:_average", "piqa_parquet"]
TXT_METRIC_PRIORITY = [
    "acc",
    "em",
    "qem",
    "exact_match",
    "quasi_exact_match",
    "f1",
    "acc_norm",
    "perplexity",
    "ppl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("evaluation/lighteval_outputs"))
    parser.add_argument("--results-json", type=Path, default=None, help="Explicit LightEval results_*.json file.")
    parser.add_argument("--model", default="", help="Original model path or HF id used for the run.")
    parser.add_argument("--display-model", default=None, help="Checkpoint label when an adapter was temporarily merged.")
    parser.add_argument("--tasks", default="", help="Task string used for the run.")
    parser.add_argument("--csv", type=Path, default=Path("evaluation/eval_results_txt/general_eval_results.csv"))
    parser.add_argument("--txt", type=Path, default=Path("evaluation/eval_results_txt/general_eval_results.txt"))
    parser.add_argument("--min-mtime", type=float, default=None, help="Only consider results modified after this epoch.")
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise ValueError(f"{path} is not a LightEval results JSON with a top-level 'results' object")
    return data


def timestamp_from_filename(path: Path) -> str:
    match = re.match(r"results_(.+)\.json$", path.name)
    return match.group(1) if match else path.stem


def model_candidates(model: str) -> set[str]:
    if not model:
        return set()
    stripped = model.strip("/")
    return {
        model,
        stripped,
        model.replace("/", "_"),
        stripped.replace("/", "_"),
    }


def path_model_name(path: Path, output_root: Path) -> str:
    try:
        rel_parent = path.parent.relative_to(output_root / "results")
    except ValueError:
        return ""
    return rel_parent.as_posix()


def matches_model(data: dict[str, Any], path: Path, output_root: Path, candidates: set[str]) -> bool:
    if not candidates:
        return True

    config_general = data.get("config_general") or {}
    result_model = str(config_general.get("model_name") or "")
    result_path_model = path_model_name(path, output_root)

    comparable = {
        result_model,
        result_model.strip("/"),
        result_model.replace("/", "_"),
        result_model.strip("/").replace("/", "_"),
        result_path_model,
        result_path_model.strip("/"),
        result_path_model.replace("/", "_"),
        result_path_model.strip("/").replace("/", "_"),
    }
    return bool(candidates.intersection(comparable))


def find_latest_result(output_root: Path, model: str, min_mtime: float | None) -> Path:
    results_root = output_root / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"LightEval results directory does not exist: {results_root}")

    candidates = model_candidates(model)
    matches: list[tuple[float, Path]] = []
    for path in results_root.rglob("results_*.json"):
        stat = path.stat()
        if min_mtime is not None and stat.st_mtime < min_mtime:
            continue
        try:
            data = load_result(path)
        except Exception:
            continue
        if matches_model(data, path, output_root, candidates):
            matches.append((stat.st_mtime, path))

    if not matches:
        model_hint = f" for model '{model}'" if model else ""
        raise FileNotFoundError(f"No LightEval results JSON found{model_hint} under {results_root}")

    matches.sort(key=lambda item: (item[0], item[1].as_posix()))
    return matches[-1][1]


def scalar_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def infer_tasks(data: dict[str, Any]) -> str:
    config_tasks = data.get("config_tasks")
    if isinstance(config_tasks, dict) and config_tasks:
        return ",".join(config_tasks.keys())
    results = data.get("results") or {}
    return ",".join(k for k in results.keys() if isinstance(k, str))


def flatten_result(data: dict[str, Any], result_json: Path, model_input: str, tasks: str) -> OrderedDict[str, Any]:
    config_general = data.get("config_general") or {}
    row: OrderedDict[str, Any] = OrderedDict()
    row["run_timestamp"] = timestamp_from_filename(result_json)
    row["model_name"] = config_general.get("model_name") or model_input
    row["model_input"] = model_input
    row["tasks"] = tasks or infer_tasks(data)
    row["result_json"] = result_json.as_posix()
    row["lighteval_sha"] = config_general.get("lighteval_sha", "")
    row["model_dtype"] = config_general.get("model_dtype", "")
    row["model_size"] = config_general.get("model_size", "")
    row["max_samples"] = config_general.get("max_samples", "")
    row["total_evaluation_time_seconds"] = (
        config_general.get("total_evaluation_time_seconds")
        or config_general.get("total_evaluation_time_secondes")
        or ""
    )
    row["exported_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for task_name, metrics in (data.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            row[f"{task_name}_{metric_name}"] = scalar_value(metric_value)

    return row


def read_existing_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def merge_fieldnames(existing: list[str], new_row: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for field in BASE_FIELDS:
        if field in existing or field in new_row:
            fields.append(field)
    for field in existing:
        if field not in fields:
            fields.append(field)
    for field in sorted(new_row):
        if field not in fields:
            fields.append(field)
    return fields


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def parse_metric_field(field: str) -> tuple[str, str] | None:
    if field in BASE_FIELDS or field.endswith("_stderr"):
        return None
    for metric in TXT_METRIC_PRIORITY:
        suffix = f"_{metric}"
        if field.endswith(suffix):
            task_name = field[: -len(suffix)]
            if task_name:
                return task_name, metric
    return None


def benchmark_name(task_name: str) -> str:
    return task_name.split("|", 1)[0]


def metric_priority(metric: str) -> int:
    try:
        return TXT_METRIC_PRIORITY.index(metric)
    except ValueError:
        return len(TXT_METRIC_PRIORITY)


def collect_txt_benchmarks(fieldnames: list[str]) -> OrderedDict[str, list[tuple[str, str, str]]]:
    benchmarks: OrderedDict[str, list[tuple[str, str, str]]] = OrderedDict()
    for field in fieldnames:
        parsed = parse_metric_field(field)
        if parsed is None:
            continue
        task_name, metric = parsed
        name = benchmark_name(task_name)
        benchmarks.setdefault(name, []).append((field, task_name, metric))

    for sources in benchmarks.values():
        sources.sort(key=lambda source: metric_priority(source[2]))

    ordered: OrderedDict[str, list[tuple[str, str, str]]] = OrderedDict()
    for name in TXT_PRIMARY_BENCHMARKS:
        ordered[name] = benchmarks.pop(name, [])
    for name, sources in benchmarks.items():
        ordered[name] = sources
    return ordered


def format_result_value(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "-"
    try:
        number = float(text)
    except ValueError:
        return text
    formatted = f"{number:.4f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def value_for_benchmark(row: dict[str, Any], sources: list[tuple[str, str, str]]) -> str:
    row_tasks = {
        task.strip()
        for task in str(row.get("tasks", "")).split(",")
        if task.strip()
    }
    preferred_sources = sorted(
        sources,
        key=lambda source: (
            source[1] not in row_tasks if row_tasks else False,
            metric_priority(source[2]),
        ),
    )
    for field, _task_name, _metric in preferred_sources:
        value = row.get(field, "")
        if str(value).strip():
            return format_result_value(value)
    return "-"


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text


def write_txt(path: Path, _csv_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    benchmarks = collect_txt_benchmarks(fieldnames)
    selected = ["model_name", *benchmarks.keys()]

    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_row = [md_escape(row.get("model_name", ""))]
        for sources in benchmarks.values():
            rendered_row.append(md_escape(value_for_benchmark(row, sources)))
        rendered_rows.append(rendered_row)

    rendered_headers = [md_escape(field) for field in selected]
    widths = [len(header) for header in rendered_headers]
    for rendered_row in rendered_rows:
        for idx, cell in enumerate(rendered_row):
            widths[idx] = max(widths[idx], len(cell))

    lines: list[str] = []
    if not selected:
        lines.append("(no results)")
    else:
        header = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(rendered_headers)) + " |"
        sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
        lines.extend([header, sep])
        for rendered_row in rendered_rows:
            lines.append("| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(rendered_row)) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_json = args.results_json or find_latest_result(args.output_root, args.model, args.min_mtime)
    result_json = result_json.resolve()
    data = load_result(result_json)
    new_row = flatten_result(data, result_json, args.model, args.tasks)
    if args.display_model:
        new_row["model_name"] = args.display_model
        new_row["model_input"] = args.display_model

    existing_fields, existing_rows = read_existing_csv(args.csv)
    fieldnames = merge_fieldnames(existing_fields, new_row)
    rows: list[dict[str, Any]] = [*existing_rows, new_row]

    write_csv(args.csv, fieldnames, rows)
    write_txt(args.txt, args.csv, fieldnames, rows)

    print(f"Exported LightEval result: {result_json}")
    print(f"Updated CSV: {args.csv}")
    print(f"Updated TXT: {args.txt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
