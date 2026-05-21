import unittest

from bg_rl.engine import BasicEngine, rankings_to_records
from bg_rl.state import BackgammonState


class EngineTests(unittest.TestCase):
    def test_basic_engine_chooses_legal_move(self) -> None:
        state = BackgammonState.initial(turn=0)
        dice = (3, 1)
        engine = BasicEngine()

        action = engine.choose_move(state, dice)

        self.assertIn(action, state.legal_actions(dice))

    def test_rankings_to_records(self) -> None:
        state = BackgammonState.initial(turn=0)
        rankings = BasicEngine().rank_moves(state, (3, 1))
        records = rankings_to_records(rankings[:2])

        self.assertEqual(len(records), 2)
        self.assertIn("action", records[0])
        self.assertIn("score", records[0])


if __name__ == "__main__":
    unittest.main()
