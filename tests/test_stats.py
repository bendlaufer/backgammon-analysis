import json
import unittest

from bg_rl.stats import merge_stats, summarize_jsonl_lines


class StatsTests(unittest.TestCase):
    def test_summarize_full_rows(self) -> None:
        row = {
            "legal_actions": [["0/1:1"], ["0/2:2"]],
            "selected_action_index": 1,
        }

        stats = summarize_jsonl_lines(
            [json.dumps(row)],
            max_legal_actions=512,
        )

        self.assertEqual(stats.rows, 1)
        self.assertEqual(stats.full_legal_labels, 1)
        self.assertEqual(stats.trainable_rows, 1)
        self.assertEqual(stats.percentile(100), 2)

    def test_merge_stats(self) -> None:
        row = {
            "legal_actions": [["0/1:1"]],
            "selected_action_index": None,
        }
        a = summarize_jsonl_lines([json.dumps(row)], max_legal_actions=512)
        b = summarize_jsonl_lines([json.dumps(row)], max_legal_actions=512)

        merged = merge_stats([a, b])

        self.assertEqual(merged.rows, 2)
        self.assertEqual(merged.full_legal_labels, 0)
        self.assertEqual(merged.partial_or_unindexed_labels, 2)


if __name__ == "__main__":
    unittest.main()
