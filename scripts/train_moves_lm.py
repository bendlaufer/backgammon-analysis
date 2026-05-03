import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Iterable
import zipfile

import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a next-token LM on backgammon moves.")
    parser.add_argument("--dataset", default="ArkadiumInc/ArkadiumBackgammon")
    parser.add_argument("--config", default="gamelogs")
    parser.add_argument(
        "--mat-zip",
        default="",
        help="Optional path to full gamelogs zip (.mat files). Uses local full corpus when set.",
    )
    parser.add_argument(
        "--data-jsonl",
        default="",
        help="Optional preprocessed JSONL with a 'moves' field per row.",
    )
    parser.add_argument("--output-dir", default="artifacts/moves-lm")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows held out and never seen during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_moves(examples: Iterable[dict]) -> Iterable[str]:
    for row in examples:
        moves = row.get("moves")
        if isinstance(moves, str) and moves.strip():
            yield moves


def build_tokenizer(train_ds: Dataset) -> PreTrainedTokenizerFast:
    counter: Counter[str] = Counter()
    for moves in iter_moves(train_ds):
        counter.update(moves.split())

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(counter.keys()):
        vocab[token] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    return fast_tokenizer


def tokenize_batch(batch: dict, tokenizer: PreTrainedTokenizerFast, seq_len: int) -> dict:
    return tokenizer(
        batch["moves"],
        truncation=True,
        max_length=seq_len,
    )


HEADER_RE = re.compile(r'^\s*;\s*\[(?P<key>[^\s]+)\s+"(?P<value>.*)"\]\s*$')
MOVE_LINE_RE = re.compile(r"^\s*\d+\)")


def parse_mat_record(text: str) -> dict:
    headers: dict[str, str] = {}
    move_lines: list[str] = []
    for line in text.splitlines():
        header_match = HEADER_RE.match(line)
        if header_match:
            headers[header_match.group("key")] = header_match.group("value")
            continue
        if MOVE_LINE_RE.match(line):
            move_lines.append(line.strip())
    return {
        "player1": headers.get("Player 1", ""),
        "player2": headers.get("Player 2", ""),
        "event_date": headers.get("EventDate", ""),
        "result": headers.get("RE", ""),
        "moves": "\n".join(move_lines).strip(),
        "raw_log": text,
    }


def load_local_zip_dataset(zip_path: str) -> Dataset:
    path = Path(zip_path).expanduser().resolve()
    rows: list[dict] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".mat"):
                continue
            with zf.open(name, "r") as fp:
                text = fp.read().decode("utf-8", errors="replace")
            row = parse_mat_record(text)
            row["file_name"] = Path(name).name
            rows.append(row)
    return Dataset.from_list(rows)


def load_raw_dataset(args: argparse.Namespace) -> Dataset:
    if args.data_jsonl:
        ds = load_dataset("json", data_files=args.data_jsonl)["train"]
    elif args.mat_zip:
        ds = load_local_zip_dataset(args.mat_zip)
    else:
        ds = load_dataset(args.dataset, args.config)["train"]
    return ds


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset(args)
    raw = raw.filter(lambda x: isinstance(x["moves"], str) and len(x["moves"].strip()) > 0)

    if args.max_samples > 0 and args.max_samples < len(raw):
        raw = raw.select(range(args.max_samples))

    split = raw.train_test_split(test_size=args.holdout_ratio, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Build vocabulary only from training rows so holdout rows stay genuinely unseen.
    tokenizer = build_tokenizer(train_ds)
    tokenizer.save_pretrained(output_dir)

    # Persist row-level holdout data for qualitative prefix completion tests.
    eval_ds.to_json(str(output_dir / "holdout_rows.jsonl"))

    train_tok = train_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.seq_len),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.seq_len),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(model_config)

    if torch.backends.mps.is_available():
        use_fp16 = False
    else:
        use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)
    print(f"Saved model and tokenizer to: {output_dir / 'final'}")
    print(f"Saved unseen holdout rows to: {output_dir / 'holdout_rows.jsonl'}")


if __name__ == "__main__":
    main()
