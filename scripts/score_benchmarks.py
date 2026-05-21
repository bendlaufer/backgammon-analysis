import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.benchmarks import load_benchmarks, score_benchmark
from bg_rl.engine import BasicEngine, engine_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score benchmark positions with the current engine.")
    parser.add_argument("--benchmarks", default="benchmarks/known_strong_positions.jsonl")
    parser.add_argument("--model-path", default="artifacts/bc-policy-full-local/model.pt")
    parser.add_argument("--out", default="artifacts/benchmark_reports/score_report.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = load_benchmarks(args.benchmarks)
    engine = BasicEngine(args.model_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for benchmark in benchmarks:
            report = score_benchmark(benchmark, engine)
            report["engine"] = engine_summary(engine)
            fp.write(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"Wrote {len(benchmarks)} benchmark scores to {out_path}")


if __name__ == "__main__":
    main()
