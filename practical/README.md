# Bilingual EN-RO Summarization Pipeline

Master-level NLP project for bilingual text summarization (English + Romanian) with:
- multilingual fine-tuning (`mT5` by default),
- semantic quality evaluation using `BERTScore`,
- factual consistency evaluation using `SummaC`.

## 1) Environment Setup (Conda)

From this folder:

```bash
conda env create -f environment.yml
conda activate nlp-summarization-en-ro

# ensure conda python is used (not Homebrew python)
export PATH="$CONDA_PREFIX/bin:$PATH"
which python

# core stack
python -m pip install -r requirements.txt

# SummaC has outdated pip constraints; install without pulling old HF deps
python -m pip install summac --no-deps
```

If `pip install summac` breaks `datasets` (ImportError on `insecure_hashlib`), repair with:

```bash
python -m pip install "huggingface-hub>=0.21.2,<1.0" "transformers==4.44.0" "tokenizers==0.19.1" protobuf
python -m pip install summac --no-deps
```

## 2) Project Structure

```text
practical/
  outputs/
  src/
    data/
    eval/
    config.py
    train.py
    infer.py
    pipeline.py
```

## 3) Dataset Contracts

### English datasets
- CNN/DailyMail and XSum are loaded directly from Hugging Face `datasets`.
- No manual download is required.

### Romanian datasets
- `readerbench/ro-text-summarization` (`Content -> source`, `Summary -> reference`)
- `MaLA-LM/MassiveSumm_long` filtered to Romanian (`text -> source`, `summary -> reference`)
- All are loaded directly from Hugging Face `datasets`.
- Missing splits are auto-generated so each dataset has `train`, `validation`, and `test`.

Note: `MaLA-LM/MassiveSumm_long` is gated on Hugging Face — accept the dataset terms and run `huggingface-cli login` before loading.

No local `data/` folder is required for the default workflow.

## 4) Training

Default training (mixed EN+RO):

```bash
python -m src.train \
  --output_dir outputs/checkpoints/mt5-small-en-ro \
  --epochs 2 \
  --train_batch_size 2 \
  --eval_batch_size 2
```

## 5) Inference

Generate summaries for evaluation split:

```bash
python -m src.infer \
  --model_dir outputs/checkpoints/mt5-small-en-ro \
  --split test \
  --output_file outputs/predictions/test_predictions.jsonl
```

## 6) Evaluation

### BERTScore

```bash
python -m src.eval.bertscore_eval \
  --predictions outputs/predictions/test_predictions.jsonl \
  --output outputs/reports/bertscore_report.json
```

### SummaC

```bash
python -m src.eval.summac_eval \
  --predictions outputs/predictions/test_predictions.jsonl \
  --output outputs/reports/summac_report.json
```

## 7) End-to-End Pipeline

Run selected stages from one entrypoint:

```bash
python -m src.pipeline --stages prepare train infer eval smoke
```

You can run only specific stages, for example:

```bash
python -m src.pipeline --stages infer eval
```

## 8) Reproducibility Notes

- Global seed is configurable from CLI/config.
- Evaluation generation uses deterministic decoding defaults.
- All outputs are versioned by output directory and timestamp-safe filenames.
- Report files are emitted as JSON and CSV.

## 9) Compute Guidance

- Start with small batches and fewer epochs for smoke validation.
- For full experiments, increase epochs and max token lengths and enable GPU.
- If memory is limited, reduce `max_source_length` and `max_target_length`.
