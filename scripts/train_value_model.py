import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from bg_rl.value import TrajectoryValueDataset, ValueNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised value baseline from trajectory outcomes.")
    parser.add_argument("--data", default="artifacts/trajectories/checker_decisions.jsonl")
    parser.add_argument("--output-dir", default="artifacts/value-model")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--target-scale", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = TrajectoryValueDataset(
        args.data,
        max_samples=args.max_samples,
        target_scale=args.target_scale,
    )
    if len(dataset) < 2:
        raise ValueError("not enough trajectory rows with terminal value targets")

    eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_size = len(dataset) - eval_size
    train_ds, eval_ds = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    model = ValueNet(hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for features, targets in train_loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        eval_loss, eval_sign_acc = evaluate(model, eval_loader, loss_fn)
        print(
            f"epoch={epoch} "
            f"train_mse={train_loss / train_batches:.4f} "
            f"eval_mse={eval_loss:.4f} "
            f"eval_sign_acc={eval_sign_acc:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "target_scale": args.target_scale,
            "max_samples": args.max_samples,
        },
        output_dir / "model.pt",
    )
    print(f"Saved value model to {output_dir / 'model.pt'}")


def evaluate(
    model: ValueNet, loader: DataLoader, loss_fn: torch.nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_sign_acc = 0.0
    batches = 0
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            total_loss += loss_fn(predictions, targets.float()).item()
            sign_matches = torch.sign(predictions) == torch.sign(targets)
            total_sign_acc += sign_matches.float().mean().item()
            batches += 1
    return total_loss / batches, total_sign_acc / batches


if __name__ == "__main__":
    main()
