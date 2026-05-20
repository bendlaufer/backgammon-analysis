import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exported trajectory JSONL rows.")
    parser.add_argument("--data", default="artifacts/trajectories/checker_decisions.jsonl")
    parser.add_argument("--max-legal-actions", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.data)
    total = 0
    full = 0
    trainable = 0
    candidate_counts: list[int] = []

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            total += 1
            candidate_count = len(record["legal_actions"])
            candidate_counts.append(candidate_count)
            if record.get("selected_action_is_full_legal_action"):
                full += 1
                if candidate_count <= args.max_legal_actions:
                    trainable += 1

    sorted_counts = sorted(candidate_counts)
    print(f"rows={total}")
    print(f"full_legal_labels={full}")
    print(f"partial_or_unindexed_labels={total - full}")
    print(f"trainable_at_max_legal_actions_{args.max_legal_actions}={trainable}")
    print(f"mean_legal_actions={mean(candidate_counts):.2f}")
    for percentile in (50, 90, 95, 99, 100):
        index = int((len(sorted_counts) - 1) * percentile / 100)
        print(f"p{percentile}_legal_actions={sorted_counts[index]}")


if __name__ == "__main__":
    main()
