# backgammon-analysis

This repository is set up to analyze publicly available datasets of backgammon games.

## Setup

Create and activate a local virtual environment (one-time):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Register the venv as a notebook kernel (one-time):

```bash
python -m ipykernel install --user --name backgammon-analysis --display-name "Python (backgammon-analysis)"
```

Or use the Makefile (optional):

```bash
make setup
```

## Dataset preview

Run the preview loader script:

```bash
python scripts/load_dataset_preview.py
```

Or with Makefile:

```bash
make preview
```

## Programmatic access to full raw data

The Hugging Face dataset loader provides only preview parquet rows in `data/`.
To access the full corpus, download the raw ZIP archive(s) from the dataset repo.

Download full non-image logs ZIP:

```bash
make download-raw
```

Or directly:

```bash
python scripts/download_raw_archives.py --which gamelogs --out-dir data
```

After download, train from full raw `.mat` logs:

```bash
python scripts/train_moves_lm.py --mat-zip "data/Arkadium_Backgammon_full_data_gamelogs_001.zip" --max-samples 0 --seq-len 96 --epochs 2 --batch-size 8 --holdout-ratio 0.2
```

## Train a next-token transformer on move sequences

This project includes a from-scratch causal language model pipeline over the dataset's `moves` column:

- `scripts/train_moves_lm.py`
- `scripts/generate_moves.py`

Tokenizer behavior:

- Vocabulary is built only from tokens observed in `moves` (plus standard special tokens `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`).
- Token splitting uses whitespace, matching move-sequence token boundaries.

Train locally:

```bash
make train-lm
```

Train from the full local non-image logs zip (recommended for real-scale training):

```bash
python scripts/train_moves_lm.py --mat-zip "data/Arkadium_Backgammon_full_data_gamelogs_001.zip" --max-samples 0 --seq-len 96 --epochs 2 --batch-size 8 --holdout-ratio 0.2
```

Run generation with a prompt:

```bash
make generate-lm
```

Custom training size/length (useful on laptops):

```bash
python scripts/train_moves_lm.py --max-samples 20000 --seq-len 96 --epochs 2 --batch-size 8 --holdout-ratio 0.2
```

Model artifacts are written under `artifacts/moves-lm/` and are git-ignored.

The trainer also writes `artifacts/moves-lm/holdout_rows.jsonl`, which contains rows never seen during training, so you can test genuine unseen-prefix completion quality.

## Notebook-first usage (recommended)

Open and run:

- `notebooks/backgammon_playground.ipynb`

In Cursor, select the kernel:

- `Python (backgammon-analysis)` (or your `.venv` interpreter)

This gives you a workbook interface for exploration while still keeping dependency tracking in git.

## Keeping package versions tracked

After installing/updating packages in your active `.venv`, refresh `requirements.txt` with exact versions:

```bash
pip freeze > requirements.txt
```

Then commit the updated `requirements.txt`.

The script loads:

- `ArkadiumInc/ArkadiumBackgammon` with config `gamelogs`
- `ArkadiumInc/ArkadiumBackgammon` with config `gamelogs_with_images`

## Data and Git tracking

Large dataset artifacts and local caches are intentionally ignored in `.gitignore` (for example `data/`, `datasets/`, `.cache/`, and large data formats like `*.parquet`).

This keeps the repository lightweight while still tracking code and reproducible dependencies in `requirements.txt`.