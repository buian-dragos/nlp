"""Inference script for summary generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import CONFIG
from src.data.loaders import build_combined_split, ensure_output_dirs, load_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summaries from a trained model.")
    parser.add_argument("--model_dir", required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=CONFIG.max_source_length)
    parser.add_argument("--max_target_length", type=int, default=CONFIG.max_target_length)
    parser.add_argument("--num_beams", type=int, default=CONFIG.num_beams)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--no_ro", action="store_true", help="Disable Romanian datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    datasets_map = load_all_datasets(include_ro=not args.no_ro)
    eval_ds = build_combined_split(
        args.split,
        datasets_map,
        limit_per_dataset=args.limit_per_dataset,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(0, len(eval_ds), args.batch_size):
        batch = eval_ds[i : i + args.batch_size]
        encoded = tokenizer(
            batch["source"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_source_length,
            padding=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
            )
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for idx, pred in enumerate(predictions):
            records.append(
                {
                    "id": batch["id"][idx],
                    "language": batch["language"][idx],
                    "dataset": batch["dataset"][idx],
                    "source": batch["source"][idx],
                    "reference": batch["reference"][idx],
                    "prediction": pred.strip(),
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} predictions to {output_path}")


if __name__ == "__main__":
    main()
