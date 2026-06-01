"""Compute BERTScore for generated summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bert_score import score as bertscore_score

from src.config import CONFIG
from src.eval.common import grouped_score_summary, load_prediction_rows, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERTScore evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--model_type",
        default=CONFIG.bertscore_model,
        help="Model used internally by BERTScore.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.predictions)
    df = pd.DataFrame(rows)
    required_columns = {"prediction", "reference", "language", "dataset", "id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

    precision, recall, f1 = bertscore_score(
        cands=df["prediction"].tolist(),
        refs=df["reference"].tolist(),
        model_type=args.model_type,
        batch_size=args.batch_size,
        verbose=True,
    )

    df["bertscore_precision"] = precision.tolist()
    df["bertscore_recall"] = recall.tolist()
    df["bertscore_f1"] = f1.tolist()

    grouped = grouped_score_summary(df, "bertscore_f1")
    output_json = Path(args.output)
    output_csv = output_json.with_suffix(".csv")
    save_report(
        output_json_path=output_json,
        output_csv_path=output_csv,
        metadata={
            "metric": "bertscore_f1",
            "model_type": args.model_type,
            "input_file": str(args.predictions),
            "num_examples": len(df),
        },
        instance_df=df[
            [
                "id",
                "language",
                "dataset",
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
            ]
        ],
        grouped_df=grouped,
    )
    print(f"BERTScore reports written to {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
