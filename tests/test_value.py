import json
import unittest

import torch

from bg_rl.value import TrajectoryValueDataset, ValueNet, value_target_from_record


class ValueModelTests(unittest.TestCase):
    def setUp(self) -> None:
        with open("artifacts/trajectories/sample_checker_decisions.jsonl", encoding="utf-8") as fp:
            self.record = json.loads(next(fp))

    def test_value_target_uses_player_perspective(self) -> None:
        self.assertEqual(value_target_from_record(self.record, target_scale=16), 0.375)

    def test_value_dataset_and_model_forward(self) -> None:
        dataset = TrajectoryValueDataset(
            "artifacts/trajectories/sample_checker_decisions.jsonl",
            max_samples=8,
        )
        features, target = dataset[0]
        model = ValueNet(hidden_dim=16)
        prediction = model(features.unsqueeze(0))

        self.assertEqual(tuple(features.shape), (36,))
        self.assertEqual(tuple(prediction.shape), (1,))
        self.assertTrue(torch.isfinite(target))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
