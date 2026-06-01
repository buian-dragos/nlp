"""Fine-tuning script for bilingual EN-RO summarization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import CONFIG
from src.data.loaders import build_combined_split, ensure_output_dirs, load_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multilingual summarization model.")
    parser.add_argument("--model_name", default=CONFIG.model_name)
    parser.add_argument("--output_dir", default=str(CONFIG.checkpoints_dir / "mt5-en-ro"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--max_source_length", type=int, default=CONFIG.max_source_length)
    parser.add_argument("--max_target_length", type=int, default=CONFIG.max_target_length)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--no_ro", action="store_true", help="Disable Romanian datasets.")
    return parser.parse_args()


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Dataset:
    def _tokenize(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["reference"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        _tokenize,
        batched=True,
        desc="Tokenizing",
    )


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    datasets_map = load_all_datasets(include_ro=not args.no_ro)
    train_ds = build_combined_split("train", datasets_map, limit_per_dataset=args.limit_per_dataset)
    valid_ds = build_combined_split("validation", datasets_map, limit_per_dataset=args.limit_per_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized_train = tokenize_dataset(
        train_ds,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    tokenized_valid = tokenize_dataset(
        valid_ds,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "limit_per_dataset": args.limit_per_dataset,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "datasets": list(datasets_map.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
