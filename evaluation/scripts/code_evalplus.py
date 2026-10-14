"""Standalone CPU bridge for EvalPlus 0.3.1; no project installation is required.

Model generation runs in the training environment. This script downloads the
benchmark tasks and evaluates generated Python with EvalPlus's execution rules.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile


VERSION = "0.3.1"
BENCHMARKS = ("humaneval", "mbpp")


def check_environment():
    try:
        version = importlib.metadata.version("evalplus")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("EvalPlus is missing. Install requirements/code_eval.txt in a CPU environment "
                           "and select its Python with --evalplus-python.") from error
    if version != VERSION:
        raise RuntimeError(f"Expected evalplus=={VERSION}, found {version}")
    # Import the actual scoring path, not only package metadata.
    from evalplus.evaluate import check_correctness, get_groundtruth  # noqa: F401
    from evalplus.data import get_human_eval_plus, get_mbpp_plus  # noqa: F401
    return version


def configure_cache(path):
    cache = Path(path).absolute()
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)


def load_problems(benchmark):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if benchmark == "humaneval" else get_mbpp_plus()


def question_for(benchmark, problem):
    if benchmark == "humaneval":
        return ("Complete the following Python function. Return the full function "
                "(signature included) inside a single ```python ... ``` block.\n\n"
                f"{problem['prompt'].rstrip()}\n")
    lines = [problem["prompt"].strip()]
    if problem.get("test_list"):
        lines += ["", "Your function must satisfy these tests:"]
        lines += [line.strip() for line in problem["test_list"]]
    lines += ["", "Return the full function inside a single ```python ... ``` block. "
              "Do not include the asserts in your answer."]
    return "\n".join(lines)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(output_dir, benchmarks, max_samples=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for benchmark in benchmarks:
        problems = load_problems(benchmark)
        chosen = sorted(problems.items())
        if max_samples is not None:
            chosen = chosen[:max_samples]
        if not chosen:
            raise ValueError(f"No tasks found for {benchmark}")
        with (output / f"{benchmark}.jsonl").open("w") as stream:
            for index, (task_id, problem) in enumerate(chosen):
                stream.write(json.dumps({"index": index, "task_id": task_id,
                    "question": question_for(benchmark, problem), "entry_point": problem["entry_point"]},
                    ensure_ascii=False) + "\n")
        print(f"Prepared {benchmark}: {len(chosen)}/{len(problems)} tasks", flush=True)


def select_samples(samples, problems, expected_ids):
    if not samples:
        raise ValueError("No code samples to evaluate")
    task_ids = [row.get("task_id") for row in samples]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Pass@1 requires exactly one sample per task")
    if set(task_ids) != set(expected_ids) or len(task_ids) != len(expected_ids):
        raise ValueError("Generated samples do not match the requested benchmark tasks")
    if not set(task_ids) <= problems.keys():
        raise ValueError("Samples contain unknown benchmark task IDs")
    if any(not isinstance(row.get("solution"), str) for row in samples):
        raise ValueError("Each code sample must contain a string solution")
    return {task_id: problems[task_id] for task_id in sorted(task_ids)}


def summarize(statuses):
    if not statuses:
        raise ValueError("EvalPlus produced no results")
    base = sum(row["base_status"] == "pass" for row in statuses)
    # EvalPlus+ includes the original tests AND the additional tests.
    plus = sum(row["base_status"] == row["plus_status"] == "pass" for row in statuses)
    count = len(statuses)
    return {"questions": count, "base_correct": base, "plus_correct": plus,
            "pass1": base / count, "plus_pass1": plus / count}


def groundtruth_cache_key(benchmark, dataset_hash, task_ids):
    # Official data hashes cover the raw benchmark; task IDs distinguish smoke
    # subsets. Deserialized MBPP inputs include sets/complex numbers, not JSON.
    payload = json.dumps([VERSION, benchmark, dataset_hash, sorted(task_ids)])
    return "pause-code-" + hashlib.sha256(payload.encode()).hexdigest()


def score(benchmark, samples_path, tasks_path, output_path, parallel=4):
    from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    samples = [json.loads(line) for line in Path(samples_path).read_text().splitlines() if line.strip()]
    tasks = [json.loads(line) for line in Path(tasks_path).read_text().splitlines() if line.strip()]
    problems = load_problems(benchmark)
    selected = select_samples(samples, problems, [row["task_id"] for row in tasks])
    # Never populate the full-dataset cache with only smoke tasks.
    dataset_hash = get_human_eval_plus_hash() if benchmark == "humaneval" else get_mbpp_plus_hash()
    cache_key = groundtruth_cache_key(benchmark, dataset_hash, selected)
    special = MBPP_OUTPUT_NOT_NONE_TASKS if benchmark == "mbpp" else []
    expected = get_groundtruth(selected, cache_key, special)
    futures = []
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        for row in samples:
            task_id = row["task_id"]
            futures.append(pool.submit(check_correctness, benchmark, 0, selected[task_id],
                row["solution"], expected[task_id], base_only=False, fast_check=True, identifier=task_id))
        statuses = [{"task_id": result["task_id"], "base_status": result["base"][0],
                     "plus_status": result["plus"][0]} for result in (future.result() for future in futures)]
    summary = summarize(statuses)
    summary.update(benchmark=benchmark, evalplus_version=VERSION, total_benchmark_tasks=len(problems),
                   subset=len(selected) != len(problems), task_ids=list(selected))
    result = {"summary": summary, "results": statuses}
    write_json(output_path, result)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "prepare", "score"])
    parser.add_argument("--cache-dir", default="artifacts/evalplus")
    parser.add_argument("--output-dir")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--samples")
    parser.add_argument("--tasks")
    parser.add_argument("--output")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    # check does not create a cache or contact the dataset server.
    if args.action != "check":
        configure_cache(args.cache_dir)
    check_environment()
    if args.action == "check":
        print(f"EvalPlus {VERSION} scoring imports passed")
    elif args.action == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output-dir")
        prepare(args.output_dir, args.benchmarks, args.max_samples)
    else:
        if not all((args.benchmark, args.samples, args.tasks, args.output)):
            parser.error("score requires --benchmark, --samples, --tasks, and --output")
        score(args.benchmark, args.samples, args.tasks, args.output, args.parallel)


if __name__ == "__main__":
    main()
