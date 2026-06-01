"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def load_prediction_rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("Prediction file is empty.")
    return rows


def grouped_score_summary(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["language", "dataset"])[score_column]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    grouped["ci95"] = grouped.apply(
        lambda row: 0.0
        if row["count"] <= 1
        else 1.96 * float(row["std"]) / math.sqrt(float(row["count"])),
        axis=1,
    )
    grouped = grouped.rename(
        columns={
            "mean": f"{score_column}_mean",
            "std": f"{score_column}_std",
            "count": "n",
            "ci95": f"{score_column}_ci95",
        }
    )
    return grouped


def save_report(
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    metadata: dict,
    instance_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
) -> None:
    output_json_path = Path(output_json_path)
    output_csv_path = Path(output_csv_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "grouped": grouped_df.to_dict(orient="records"),
        "instances": instance_df.to_dict(orient="records"),
    }
    output_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    grouped_df.to_csv(output_csv_path, index=False)
