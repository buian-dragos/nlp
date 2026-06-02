"""Centralized configuration for the EN-RO summarization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    outputs_dir: Path = project_root / "outputs"

    reports_dir: Path = outputs_dir / "reports"
    predictions_dir: Path = outputs_dir / "predictions"
    checkpoints_dir: Path = outputs_dir / "checkpoints"

    model_name: str = "google/mt5-small"
    bertscore_model: str = "xlm-roberta-large"
    seed: int = 42

    max_source_length: int = 512
    max_target_length: int = 128
    num_beams: int = 4


CONFIG = ProjectConfig()
