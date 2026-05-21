import argparse
import glob
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from bg_rl.bc import BC_FEATURE_DIM, STATE_FEATURE_DIM, encode_action_tokens, encode_dice, encode_state_record
from bg_rl.rl import PolicyValueNet, parse_rl_record


class RLReplayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, paths: list[str], *, max_samples: int = 0, max_legal_actions: int = 512) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for path in expand_paths(paths):
            with Path(path).open("r", encoding="utf-8") as fp:
                for line in fp:
                    record = parse_rl_record(line)
                    if len(record["legal_actions"]) > max_legal_actions:
                        continue
                    self.samples.append(record_to_sample(record))
                    if max_samples > 0 and len(self.samples) >= max_samples:
                        return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy/value network from RL replay shards.")
    parser.add_argument("--data", nargs="+", default=["artifacts/rl-self-play/*.jsonl"])
    parser.add_argument("--output-dir", default="artifacts/rl-policy-value")
    parser.add_argument("--init-model", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-legal-actions", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = RLReplayDataset(
        args.data,
        max_samples=args.max_samples,
        max_legal_actions=args.max_legal_actions,
    )
    if len(dataset) < 2:
        raise ValueError("not enough RL replay rows to train")

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
        collate_fn=collate_rl_samples,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_rl_samples,
        num_workers=args.num_workers,
    )

    model = PolicyValueNet(hidden_dim=args.hidden_dim)
    if args.init_model and Path(args.init_model).exists():
        checkpoint = torch.load(args.init_model, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            loss, _metrics = loss_and_metrics(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        eval_loss, eval_metrics = evaluate(model, eval_loader)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss / train_batches:.4f} "
            f"eval_loss={eval_loss:.4f} "
            f"eval_policy_kl={eval_metrics['policy_kl']:.4f} "
            f"eval_value_mse={eval_metrics['value_mse']:.4f}"
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
    print(f"Saved RL policy/value model to {output_dir / 'model.pt'}")


def expand_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for path in paths:
        matches = sorted(glob.glob(path))
        expanded.extend(matches or [path])
    return expanded


def record_to_sample(record: dict) -> dict[str, torch.Tensor]:
    state_features = encode_state_record(record["state"])
    dice_features = encode_dice(record["dice"])
    candidate_rows = [
        torch.cat([state_features, dice_features, encode_action_tokens(action)])
        for action in record["legal_actions"]
    ]
    return {
        "candidate_features": torch.stack(candidate_rows),
        "state_features": state_features,
        "policy": torch.tensor(record["search_policy"], dtype=torch.float32),
        "value": torch.tensor(float(record["reward"]), dtype=torch.float32),
    }


def collate_rl_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_candidates = max(sample["candidate_features"].shape[0] for sample in samples)
    features = torch.zeros(batch_size, max_candidates, BC_FEATURE_DIM, dtype=torch.float32)
    policy = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    state_features = torch.zeros(batch_size, STATE_FEATURE_DIM, dtype=torch.float32)
    values = torch.zeros(batch_size, dtype=torch.float32)
    for row, sample in enumerate(samples):
        count = sample["candidate_features"].shape[0]
        features[row, :count] = sample["candidate_features"]
        policy[row, :count] = sample["policy"]
        mask[row, :count] = True
        state_features[row] = sample["state_features"]
        values[row] = sample["value"]
    return {
        "features": features,
        "policy": policy,
        "mask": mask,
        "state_features": state_features,
        "values": values,
    }


def loss_and_metrics(model: PolicyValueNet, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model.policy_logits(batch["features"])
    masked_logits = logits.masked_fill(~batch["mask"], -torch.inf)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1).masked_fill(~batch["mask"], 0.0)
    policy_loss = -(batch["policy"] * log_probs).sum(dim=1).mean()
    value_predictions = model.value_estimate(batch["state_features"])
    value_loss = torch.nn.functional.mse_loss(value_predictions, batch["values"])
    loss = policy_loss + value_loss
    return loss, {"policy_kl": float(policy_loss.item()), "value_mse": float(value_loss.item())}


def evaluate(model: PolicyValueNet, loader: DataLoader) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_policy_kl = 0.0
    total_value_mse = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, metrics = loss_and_metrics(model, batch)
            total_loss += loss.item()
            total_policy_kl += metrics["policy_kl"]
            total_value_mse += metrics["value_mse"]
            batches += 1
    return total_loss / batches, {
        "policy_kl": total_policy_kl / batches,
        "value_mse": total_value_mse / batches,
    }


if __name__ == "__main__":
    main()
