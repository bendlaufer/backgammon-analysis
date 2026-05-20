import json
import unittest
import zipfile

import torch

from bg_rl.bc import (
    BC_FEATURE_DIM,
    CandidatePolicyNet,
    TrajectoryBCDataset,
    collate_bc_samples,
    encode_action_tokens,
    encode_dice,
    encode_state_record,
    masked_accuracy,
    masked_cross_entropy,
    record_to_bc_sample,
)
from bg_rl.mat import parse_match_text
from bg_rl.trajectory import match_to_compact_records


class BehaviorCloningTests(unittest.TestCase):
    def setUp(self) -> None:
        with open("artifacts/trajectories/sample_checker_decisions.jsonl", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                if record["selected_action_is_full_legal_action"]:
                    self.record = record
                    return
        raise AssertionError("sample trajectory file has no full-legal rows")

    def test_record_to_bc_sample_aligns_label(self) -> None:
        sample = record_to_bc_sample(self.record)

        self.assertEqual(sample.candidate_features.shape[1], BC_FEATURE_DIM)
        self.assertEqual(sample.selected_index, self.record["selected_action_index"])

    def test_compact_record_to_bc_sample_regenerates_label(self) -> None:
        with zipfile.ZipFile("data/Arkadium_Backgammon_full_data_gamelogs_001.zip") as zf:
            text = zf.read("1200001.mat").decode("utf-8")
        compact = match_to_compact_records(parse_match_text(text), source_file="1200001.mat")[0]

        sample = record_to_bc_sample(compact)

        self.assertEqual(sample.selected_index, 2)
        self.assertEqual(sample.candidate_features.shape[1], BC_FEATURE_DIM)

    def test_feature_encoders_have_stable_shapes(self) -> None:
        self.assertEqual(tuple(encode_state_record(self.record["state"]).shape), (36,))
        self.assertEqual(tuple(encode_dice(self.record["dice"]).shape), (12,))
        self.assertEqual(tuple(encode_action_tokens(self.record["selected_action"]).shape), (24,))

    def test_collate_and_model_forward(self) -> None:
        dataset = TrajectoryBCDataset(
            "artifacts/trajectories/sample_checker_decisions.jsonl",
            max_samples=4,
            max_legal_actions=512,
        )
        batch = collate_bc_samples([dataset[0], dataset[1]])
        model = CandidatePolicyNet(hidden_dim=16)
        logits = model(batch["features"])

        self.assertEqual(logits.shape, batch["mask"].shape)
        loss = masked_cross_entropy(logits, batch["mask"], batch["labels"])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(masked_accuracy(logits, batch["mask"], batch["labels"]), 0.0)


if __name__ == "__main__":
    unittest.main()
