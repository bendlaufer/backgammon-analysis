import unittest

from bg_rl.afterstate import legal_afterstates, selected_afterstate
from bg_rl.state import BackgammonState, MoveStep


class AfterstateTests(unittest.TestCase):
    def test_legal_afterstates_align_with_legal_actions(self) -> None:
        state = BackgammonState.initial(turn=0)
        actions = state.legal_actions((3, 1))
        candidates = legal_afterstates(state, (3, 1))

        self.assertEqual(len(candidates), len(actions))
        self.assertEqual(tuple(candidate.action for candidate in candidates), actions)
        self.assertTrue(all(candidate.afterstate.turn == 1 for candidate in candidates))

    def test_selected_afterstate_applies_action(self) -> None:
        state = BackgammonState.initial(turn=0)
        action = (MoveStep(7, 4, 3, False), MoveStep(5, 4, 1, False))

        self.assertEqual(selected_afterstate(state, action), state.apply_action(action))


if __name__ == "__main__":
    unittest.main()
