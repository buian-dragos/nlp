"""Minimal smoke checks for EN-RO data and tokenization pipeline."""

from __future__ import annotations

import argparse
from collections import Counter

from transformers import AutoTokenizer

from src.config import CONFIG
from src.data.loaders import build_combined_split, load_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for data/tokenization.")
    parser.add_argument("--model_name", default=CONFIG.model_name)
    parser.add_argument("--limit_per_dataset", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=CONFIG.max_source_length)
    parser.add_argument("--no_ro", action="store_true", help="Disable Romanian datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_map = load_all_datasets(include_ro=not args.no_ro)
    smoke_ds = build_combined_split(
        "train",
        datasets_map,
        limit_per_dataset=args.limit_per_dataset,
    )
    language_counts = Counter(smoke_ds["language"])
    if not args.no_ro and "ro" not in language_counts:
        raise RuntimeError("Smoke test failed: no Romanian examples found.")
    if "en" not in language_counts:
        raise RuntimeError("Smoke test failed: no English examples found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    for lang in ("en", "ro"):
        if lang not in language_counts:
            continue
        examples = [row["source"] for row in smoke_ds if row["language"] == lang][:2]
        if not examples:
            raise RuntimeError(f"Smoke test failed: no batch for language={lang}")
        encoded = tokenizer(
            examples,
            truncation=True,
            max_length=args.max_source_length,
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[0] == 0:
            raise RuntimeError(f"Smoke test failed: empty tokenized batch for {lang}")

    print("Smoke test passed: EN/RO mini-batches are valid.")


if __name__ == "__main__":
    main()
