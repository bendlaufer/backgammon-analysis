import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.benchmarks import load_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark position JSONL files.")
    parser.add_argument("path", nargs="?", default="benchmarks/known_strong_positions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = load_benchmarks(args.path)
    scoreable = sum(1 for benchmark in benchmarks if benchmark.state is not None and benchmark.dice is not None)
    print(f"benchmarks={len(benchmarks)} scoreable={scoreable} path={args.path}")


if __name__ == "__main__":
    main()
