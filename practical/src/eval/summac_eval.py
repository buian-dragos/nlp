"""Compute SummaC factual consistency scores for generated summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.eval.common import grouped_score_summary, load_prediction_rows, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SummaC evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary factuality flag.",
    )
    parser.add_argument(
        "--model_name",
        default="vitc",
        help="NLI backbone name used by SummaCZS.",
    )
    return parser.parse_args()


def _load_summac_model(model_name: str):
    try:
        from summac.model_summac import SummaCZS
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import SummaC. Check that the 'summac' package is installed."
        ) from exc

    return SummaCZS(
        granularity="sentence",
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.predictions)
    df = pd.DataFrame(rows)
    required_columns = {"source", "prediction", "language", "dataset", "id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

    model = _load_summac_model(args.model_name)
    result = model.score(df["source"].tolist(), df["prediction"].tolist())
    if "scores" not in result:
        raise ValueError("SummaC output did not include 'scores'.")

    df["summac_score"] = result["scores"]
    df["factual_flag"] = (df["summac_score"] >= args.threshold).astype(int)
    grouped = grouped_score_summary(df, "summac_score")

    output_json = Path(args.output)
    output_csv = output_json.with_suffix(".csv")
    save_report(
        output_json_path=output_json,
        output_csv_path=output_csv,
        metadata={
            "metric": "summac_score",
            "threshold": args.threshold,
            "model_name": args.model_name,
            "input_file": str(args.predictions),
            "num_examples": len(df),
            "factual_rate": float(df["factual_flag"].mean()),
        },
        instance_df=df[
            ["id", "language", "dataset", "summac_score", "factual_flag"]
        ],
        grouped_df=grouped,
    )
    print(f"SummaC reports written to {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
