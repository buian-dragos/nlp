"""End-to-end orchestration for EN-RO summarization workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import CONFIG
from src.data.loaders import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prepare/train/infer/eval/smoke stages.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["prepare", "smoke"],
        choices=["prepare", "train", "infer", "eval", "smoke"],
    )
    parser.add_argument("--model_name", default=CONFIG.model_name)
    parser.add_argument("--model_dir", default=str(CONFIG.checkpoints_dir / "mt5-small-en-ro"))
    parser.add_argument(
        "--predictions_file",
        default=str(CONFIG.predictions_dir / "test_predictions.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--infer_batch_size", type=int, default=4)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--no_ro", action="store_true", help="Disable Romanian datasets.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    model_dir = Path(args.model_dir)
    reports_dir = CONFIG.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    if "prepare" in args.stages:
        ensure_output_dirs()
        print("Prepare stage complete.")

    if "train" in args.stages:
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--model_name",
            args.model_name,
            "--output_dir",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--train_batch_size",
            str(args.train_batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
        ]
        if args.limit_per_dataset is not None:
            cmd.extend(["--limit_per_dataset", str(args.limit_per_dataset)])
        if args.no_ro:
            cmd.append("--no_ro")
        _run(cmd)

    if "infer" in args.stages:
        cmd = [
            sys.executable,
            "-m",
            "src.infer",
            "--model_dir",
            str(model_dir),
            "--split",
            args.split,
            "--output_file",
            str(args.predictions_file),
            "--batch_size",
            str(args.infer_batch_size),
        ]
        if args.limit_per_dataset is not None:
            cmd.extend(["--limit_per_dataset", str(args.limit_per_dataset)])
        if args.no_ro:
            cmd.append("--no_ro")
        _run(cmd)

    if "eval" in args.stages:
        _run(
            [
                sys.executable,
                "-m",
                "src.eval.bertscore_eval",
                "--predictions",
                str(args.predictions_file),
                "--output",
                str(reports_dir / "bertscore_report.json"),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "src.eval.summac_eval",
                "--predictions",
                str(args.predictions_file),
                "--output",
                str(reports_dir / "summac_report.json"),
            ]
        )

    if "smoke" in args.stages:
        cmd = [
            sys.executable,
            "-m",
            "src.smoke_test",
            "--model_name",
            args.model_name,
        ]
        if args.limit_per_dataset is not None:
            cmd.extend(["--limit_per_dataset", str(args.limit_per_dataset)])
        if args.no_ro:
            cmd.append("--no_ro")
        _run(cmd)

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
