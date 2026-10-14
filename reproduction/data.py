"""Load local or Hugging Face math data without changing its row order."""
from __future__ import annotations

from pathlib import Path


def load_source(source, *, split="train", config=None, revision=None, max_samples=None):
    """Return a dataset in source order, optionally limited to its first N rows."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("--max-samples must be a positive integer.")
    if not source:
        raise ValueError("A dataset path or Hugging Face dataset ID is required.")
    path = Path(source).expanduser()
    file_formats = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet", ".arrow": "arrow"}
    if path.is_file():
        if config is not None or revision is not None:
            raise ValueError("--dataset-config and --dataset-revision apply to Hugging Face datasets, not local files.")
        if split != "train":
            raise ValueError("A local dataset file has only the 'train' split; use --split train.")
        if path.suffix.lower() not in file_formats:
            raise ValueError(f"Unsupported dataset file format: {path.suffix}. Use JSONL, JSON, CSV, Parquet, or Arrow.")
        if path.stat().st_size == 0:
            raise ValueError(f"The dataset file is empty: {path}")
        try:
            data = load_dataset(file_formats[path.suffix.lower()], data_files=str(path.resolve()), split="train")
        except StopIteration as exc:
            raise ValueError(f"The dataset file contains no records: {path}") from exc
    elif path.is_dir():
        if config is not None or revision is not None:
            raise ValueError("--dataset-config and --dataset-revision apply only to Hugging Face datasets.")
        data = load_from_disk(str(path))
        if isinstance(data, DatasetDict):
            if split not in data:
                raise ValueError(f"Dataset split {split!r} was not found. Available splits: {', '.join(data)}.")
            data = data[split]
        elif split != "train":
            raise ValueError("A saved Dataset has only the 'train' split.")
    else:
        if path.suffix.lower() in file_formats or str(source).startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(f"Dataset file does not exist: {source}")
        data = load_dataset(str(source), name=config, revision=revision, split=split)
    if not isinstance(data, Dataset):
        raise ValueError("Select a single dataset split with --split.")
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    if not len(data):
        raise ValueError("The selected dataset is empty.")
    return data


def validate_columns(dataset, input_key, output_key):
    """Validate math question/solution or question/final-answer string columns."""
    missing = [key for key in dict.fromkeys((input_key, output_key)) if key not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing dataset column(s): {', '.join(missing)}. Available columns: {', '.join(dataset.column_names)}.")
    if not len(dataset):
        raise ValueError("The selected dataset is empty.")
    offset = 0
    for batch in dataset.iter(batch_size=1024):
        for key in dict.fromkeys((input_key, output_key)):
            for index, value in enumerate(batch[key]):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Dataset row {offset + index}, column {key!r} must be a non-empty string.")
        offset += len(batch[input_key])
    return len(dataset)
