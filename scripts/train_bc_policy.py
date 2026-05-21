import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from bg_rl.bc import (
    CandidatePolicyNet,
    TrajectoryBCDataset,
    collate_bc_samples,
    masked_accuracy,
    masked_cross_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior-cloning legal-action scorer.")
    parser.add_argument("--data", default="artifacts/trajectories/checker_decisions.jsonl")
    parser.add_argument("--output-dir", default="artifacts/bc-policy")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-legal-actions", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = TrajectoryBCDataset(
        args.data,
        max_samples=args.max_samples,
        max_legal_actions=args.max_legal_actions,
    )
    if len(dataset) < 2:
        raise ValueError("not enough full-legal trajectory rows to train")

    eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_size = len(dataset) - eval_size
    train_ds, eval_ds = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_bc_samples,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_bc_samples,
        num_workers=args.num_workers,
    )

    model = CandidatePolicyNet(hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        for batch in train_loader:
            logits = model(batch["features"])
            loss = masked_cross_entropy(logits, batch["mask"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += masked_accuracy(logits.detach(), batch["mask"], batch["labels"])
            train_batches += 1

        eval_loss, eval_acc = evaluate(model, eval_loader)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss / train_batches:.4f} "
            f"train_acc={train_acc / train_batches:.4f} "
            f"eval_loss={eval_loss:.4f} "
            f"eval_acc={eval_acc:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "max_legal_actions": args.max_legal_actions,
            "max_samples": args.max_samples,
        },
        output_dir / "model.pt",
    )
    print(f"Saved behavior-cloning policy to {output_dir / 'model.pt'}")


def evaluate(model: CandidatePolicyNet, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"])
            total_loss += masked_cross_entropy(logits, batch["mask"], batch["labels"]).item()
            total_acc += masked_accuracy(logits, batch["mask"], batch["labels"])
            batches += 1
    return total_loss / batches, total_acc / batches


if __name__ == "__main__":
    main()
