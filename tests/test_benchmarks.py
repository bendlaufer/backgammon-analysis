import json
import tempfile
import unittest
from pathlib import Path

from bg_rl.benchmarks import load_benchmarks, score_benchmark
from bg_rl.engine import BasicEngine


class BenchmarkSuiteTests(unittest.TestCase):
    def test_known_strong_positions_load(self) -> None:
        benchmarks = load_benchmarks("benchmarks/known_strong_positions.jsonl")

        self.assertGreaterEqual(len(benchmarks), 4)
        self.assertIn("xg_backgame_41_hit16", {benchmark.id for benchmark in benchmarks})

    def test_active_position_scores_with_engine(self) -> None:
        benchmarks = load_benchmarks("benchmarks/known_strong_positions.jsonl")
        active = next(benchmark for benchmark in benchmarks if benchmark.id == "opening_31_smoke")

        report = score_benchmark(active, BasicEngine())

        self.assertTrue(report["scoreable"])
        self.assertGreater(report["legal_action_count"], 0)
        self.assertIn("engine_top_key", report)

    def test_missing_source_is_rejected(self) -> None:
        record = {
            "id": "bad",
            "title": "Bad benchmark",
            "category": "test",
            "status": "candidate",
            "position": {"reference": "none"},
            "decision": {"kind": "mixed"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_benchmarks(path)


if __name__ == "__main__":
    unittest.main()
