import json
import random
import unittest

from bg_rl.cube import DoubleDecision, TakeDecision
from bg_rl.self_play import RandomPolicy, play_game, self_play_record_to_json


class AlwaysDoublePassPolicy:
    def choose_double(self, state):
        return DoubleDecision.DOUBLE

    def choose_take(self, offer):
        return TakeDecision.PASS


class SelfPlayTests(unittest.TestCase):
    def test_random_self_play_records_checker_rewards(self) -> None:
        game = play_game(
            game_index=7,
            policy0=RandomPolicy(random.Random(1)),
            policy1=RandomPolicy(random.Random(2)),
            rng=random.Random(3),
            max_plies=400,
        )

        self.assertIsNotNone(game.winner)
        self.assertGreater(game.points_won, 0)
        self.assertTrue(all(decision.reward in (-1.0, 1.0) for decision in game.decisions))

    def test_cube_pass_can_end_game_before_checker_move(self) -> None:
        game = play_game(
            game_index=1,
            cube_policy0=AlwaysDoublePassPolicy(),
            cube_policy1=AlwaysDoublePassPolicy(),
            rng=random.Random(4),
            max_plies=10,
        )

        self.assertEqual(game.winner, 0)
        self.assertEqual(game.points_won, 1)
        self.assertEqual([decision.phase for decision in game.decisions], ["double", "take"])

    def test_self_play_record_serializes_phase_fields(self) -> None:
        game = play_game(
            game_index=1,
            cube_policy0=AlwaysDoublePassPolicy(),
            cube_policy1=AlwaysDoublePassPolicy(),
            rng=random.Random(4),
            max_plies=10,
        )

        record = json.loads(self_play_record_to_json(game.decisions[0]))

        self.assertEqual(record["format"], "self_play_decision_v2")
        self.assertEqual(record["phase"], "double")
        self.assertEqual(record["double_decision"], "double")


if __name__ == "__main__":
    unittest.main()
