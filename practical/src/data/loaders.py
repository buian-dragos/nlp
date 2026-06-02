"""Dataset loading and normalization helpers."""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from src.config import CONFIG


def _normalize_record(
    *,
    record_id: str,
    source: str,
    reference: str,
    language: str,
    dataset_name: str,
) -> dict:
    return {
        "id": record_id,
        "language": language,
        "dataset": dataset_name,
        "source": source.strip(),
        "reference": reference.strip(),
    }


def _with_standard_splits(raw: DatasetDict, dataset_name: str, seed: int = CONFIG.seed) -> DatasetDict:
    """
    Ensure train/validation/test exist for a DatasetDict.

    Strategy:
      - If all splits exist, keep as-is.
      - If train+test exist, create validation from train (10%).
      - If only train exists, split train into train/validation/test (80/10/10).
    """
    split_names = set(raw.keys())
    if {"train", "validation", "test"}.issubset(split_names):
        return DatasetDict(
            {
                "train": raw["train"],
                "validation": raw["validation"],
                "test": raw["test"],
            }
        )

    if "train" not in split_names:
        raise ValueError(f"{dataset_name} must include at least a 'train' split.")

    if "test" in split_names:
        first_split = raw["train"].train_test_split(test_size=0.1, seed=seed)
        return DatasetDict(
            {
                "train": first_split["train"],
                "validation": first_split["test"],
                "test": raw["test"],
            }
        )

    # only train exists -> create train/validation/test
    first_split = raw["train"].train_test_split(test_size=0.2, seed=seed)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )


def load_ro_readerbench_dataset() -> DatasetDict:
    """Load and normalize readerbench/ro-text-summarization."""
    raw = load_dataset("readerbench/ro-text-summarization")
    standardized = _with_standard_splits(raw, dataset_name="ro_readerbench")

    def mapper(example, idx):
        return _normalize_record(
            record_id=f"ro_readerbench_{idx}",
            source=example["Content"],
            reference=example["Summary"],
            language="ro",
            dataset_name="ro_readerbench",
        )

    normalized = {}
    for split in ("train", "validation", "test"):
        normalized[split] = standardized[split].map(
            mapper,
            with_indices=True,
            remove_columns=standardized[split].column_names,
            desc=f"Normalizing ro_readerbench-{split}",
        )
    return DatasetDict(normalized)


# def load_ro_rolargesum_dataset() -> DatasetDict:
#     """Load and normalize avramandrei/rolargesum (gated; requires HF access approval)."""
#     raw = load_dataset("avramandrei/rolargesum")
#     standardized = _with_standard_splits(raw, dataset_name="ro_rolargesum")
#
#     def mapper(example, idx):
#         return _normalize_record(
#             record_id=f"ro_rolargesum_{idx}",
#             source=example["text"],
#             reference=example["summary"],
#             language="ro",
#             dataset_name="ro_rolargesum",
#         )
#
#     normalized = {}
#     for split in ("train", "validation", "test"):
#         normalized[split] = standardized[split].map(
#             mapper,
#             with_indices=True,
#             remove_columns=standardized[split].column_names,
#             desc=f"Normalizing ro_rolargesum-{split}",
#         )
#     return DatasetDict(normalized)


def _is_romanian(example: dict) -> bool:
    language = str(example.get("language", "")).strip().lower()
    return language in {"romanian", "ro", "ron"}


def load_ro_massivesumm_dataset() -> DatasetDict:
    """Load and normalize Romanian rows from MaLA-LM/MassiveSumm_long."""
    raw = load_dataset("MaLA-LM/MassiveSumm_long")
    ro_only = {}
    for split_name, split_data in raw.items():
        ro_only[split_name] = split_data.filter(
            _is_romanian,
            desc=f"Filtering MassiveSumm_long-{split_name} for Romanian",
            load_from_cache_file=False,
        )
    standardized = _with_standard_splits(DatasetDict(ro_only), dataset_name="ro_massivesumm")

    def mapper(example, idx):
        record_id = example.get("url") or f"ro_massivesumm_{idx}"
        return _normalize_record(
            record_id=str(record_id),
            source=example["text"],
            reference=example["summary"],
            language="ro",
            dataset_name="ro_massivesumm",
        )

    normalized = {}
    for split in ("train", "validation", "test"):
        normalized[split] = standardized[split].map(
            mapper,
            with_indices=True,
            remove_columns=standardized[split].column_names,
            desc=f"Normalizing ro_massivesumm-{split}",
        )
    return DatasetDict(normalized)


def load_en_dataset(dataset_name: str) -> DatasetDict:
    """
    Load and normalize an English summarization dataset.

    Supported names:
      - cnndm
      - xsum
    """
    if dataset_name == "cnndm":
        raw = load_dataset("cnn_dailymail", "3.0.0")

        def mapper(example, idx):
            return _normalize_record(
                record_id=f"cnndm_{idx}",
                source=example["article"],
                reference=example["highlights"],
                language="en",
                dataset_name="cnndm",
            )

    elif dataset_name == "xsum":
        raw = load_dataset("xsum")

        def mapper(example, idx):
            return _normalize_record(
                record_id=f"xsum_{idx}",
                source=example["document"],
                reference=example["summary"],
                language="en",
                dataset_name="xsum",
            )

    else:
        raise ValueError(f"Unsupported English dataset: {dataset_name}")

    normalized = {}
    for split in ("train", "validation", "test"):
        normalized[split] = raw[split].map(
            mapper,
            with_indices=True,
            remove_columns=raw[split].column_names,
            desc=f"Normalizing {dataset_name}-{split}",
        )
    return DatasetDict(normalized)


def load_all_datasets(include_ro: bool = True) -> dict[str, DatasetDict]:
    datasets_map = {
        "cnndm": load_en_dataset("cnndm"),
        "xsum": load_en_dataset("xsum"),
    }
    if include_ro:
        datasets_map["ro_readerbench"] = load_ro_readerbench_dataset()
        datasets_map["ro_massivesumm"] = load_ro_massivesumm_dataset()
    return datasets_map


def build_combined_split(
    split: str,
    datasets_map: dict[str, DatasetDict],
    limit_per_dataset: Optional[int] = None,
) -> Dataset:
    """
    Merge selected split from all datasets into one Dataset.
    """
    chunks = []
    for _, ds_dict in datasets_map.items():
        current = ds_dict[split]
        if limit_per_dataset is not None:
            current = current.select(range(min(limit_per_dataset, len(current))))
        chunks.append(current)
    if not chunks:
        raise ValueError("No datasets available for combination.")
    return concatenate_datasets(chunks).shuffle(seed=CONFIG.seed)


def ensure_output_dirs() -> None:
    for path in (
        CONFIG.outputs_dir,
        CONFIG.reports_dir,
        CONFIG.predictions_dir,
        CONFIG.checkpoints_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
