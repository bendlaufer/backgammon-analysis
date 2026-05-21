import json
import random
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from bg_rl.rl import (
    PolicyValueNet,
    SearchConfig,
    SearchMovePolicy,
    rl_record_to_json,
    search_checker_move,
)
from bg_rl.self_play import play_game
from bg_rl.state import BackgammonState
from scripts.train_rl_policy_value import RLReplayDataset, collate_rl_samples, loss_and_metrics


class RLScaffoldTests(unittest.TestCase):
    def test_search_returns_policy_over_legal_actions(self) -> None:
        state = BackgammonState.initial(turn=0)
        model = PolicyValueNet(hidden_dim=16)

        result = search_checker_move(
            state,
            (3, 5),
            model,
            rng=random.Random(1),
            config=SearchConfig(temperature=1.0),
        )

        self.assertEqual(len(result.legal_actions), len(result.action_probabilities))
        self.assertAlmostEqual(sum(result.action_probabilities), 1.0, places=5)
        self.assertIn(result.selected_action, result.legal_actions)

    def test_replay_record_trains_one_batch(self) -> None:
        state = BackgammonState.initial(turn=0)
        model = PolicyValueNet(hidden_dim=16)
        result = search_checker_move(state, (3, 5), model, rng=random.Random(1))
        line = rl_record_to_json(
            game_index=0,
            ply=0,
            player=0,
            state=state,
            dice=(3, 5),
            result=result,
            reward=1.0,
            points_reward=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.jsonl"
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            dataset = RLReplayDataset([str(path)])
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_rl_samples)))
            loss, metrics = loss_and_metrics(model, batch)

        self.assertTrue(loss.isfinite())
        self.assertGreater(metrics["policy_kl"], 0.0)
        self.assertGreaterEqual(metrics["value_mse"], 0.0)

    def test_search_policy_collects_game_history(self) -> None:
        model = PolicyValueNet(hidden_dim=16)
        policy = SearchMovePolicy(model, rng=random.Random(2))

        game = play_game(
            game_index=0,
            policy0=policy,
            policy1=policy,
            rng=random.Random(3),
            max_plies=4,
        )

        checker_decisions = [decision for decision in game.decisions if decision.phase == "checker"]
        self.assertEqual(len(policy.history), len(checker_decisions))
        self.assertGreater(len(policy.history), 0)

    def test_record_json_has_expected_format(self) -> None:
        state = BackgammonState.initial(turn=0)
        model = PolicyValueNet(hidden_dim=16)
        result = search_checker_move(state, (3, 5), model, rng=random.Random(1))
        record = json.loads(
            rl_record_to_json(
                game_index=0,
                ply=0,
                player=0,
                state=state,
                dice=(3, 5),
                result=result,
                reward=1.0,
                points_reward=1,
            )
        )

        self.assertEqual(record["format"], "rl_checker_decision_v1")
        self.assertEqual(len(record["legal_actions"]), len(record["search_policy"]))


if __name__ == "__main__":
    unittest.main()
