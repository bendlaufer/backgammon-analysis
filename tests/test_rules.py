import unittest

from bg_rl.encoding import encode_state
from bg_rl.state import BackgammonState, MoveStep


class LegalActionTests(unittest.TestCase):
    def test_initial_position_generates_legal_opening_moves(self) -> None:
        state = BackgammonState.initial(turn=0)
        actions = state.legal_actions((3, 1))

        self.assertTrue(actions)
        self.assertTrue(all(len(action) == 2 for action in actions))
        self.assertIn(
            (MoveStep(7, 4, 3, False), MoveStep(5, 4, 1, False)),
            actions,
        )

    def test_bar_checker_must_enter_before_other_moves(self) -> None:
        points = [0] * 24
        points[0] = 14
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[5] = -5
        state = BackgammonState(points=tuple(points), bar=(1, 0), turn=0)

        actions = state.legal_actions((1, 6))

        self.assertTrue(actions)
        self.assertTrue(all(action[0].from_point is None for action in actions))

    def test_blocked_bar_entry_can_force_pass(self) -> None:
        points = [0] * 24
        points[0] = 14
        points[18] = -2
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[5] = -3
        state = BackgammonState(points=tuple(points), bar=(1, 0), turn=0)

        self.assertEqual(state.legal_actions((6, 6)), ((),))

    def test_hit_moves_opponent_checker_to_bar(self) -> None:
        points = [0] * 24
        points[5] = 15
        points[3] = -1
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[6] = -4
        state = BackgammonState(points=tuple(points), turn=0)

        next_state = state.apply_action((MoveStep(5, 3, 2, True),))

        self.assertEqual(next_state.points[3], 1)
        self.assertEqual(next_state.bar, (0, 1))
        self.assertEqual(next_state.turn, 1)

    def test_bear_off_with_oversized_die_only_from_farthest_checker(self) -> None:
        points = [0] * 24
        points[0] = 14
        points[2] = 1
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[5] = -5
        state = BackgammonState(points=tuple(points), turn=0)

        actions = state.legal_actions((6, 1))

        self.assertTrue(any(MoveStep(2, None, 6, False) in action for action in actions))
        self.assertFalse(any(MoveStep(0, None, 6, False) in action for action in actions))

    def test_must_play_larger_die_when_only_one_die_can_be_used(self) -> None:
        points = [0] * 24
        points[0] = 1
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[6] = -5
        state = BackgammonState(points=tuple(points), off=(14, 0), turn=0)

        actions = state.legal_actions((6, 1))

        self.assertTrue(actions)
        self.assertTrue(all(action == (MoveStep(0, None, 6, False),) for action in actions))


class EncodingTests(unittest.TestCase):
    def test_encode_state_has_stable_shape(self) -> None:
        encoded = encode_state(BackgammonState.initial())

        self.assertEqual(tuple(encoded.shape), (36,))


if __name__ == "__main__":
    unittest.main()
