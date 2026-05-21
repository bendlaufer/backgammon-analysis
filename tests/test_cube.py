import unittest

from bg_rl.cube import (
    CubeOffer,
    DoubleDecision,
    TakeDecision,
    apply_pass,
    apply_take,
    can_double,
    checker_points_won,
    make_offer,
)
from bg_rl.state import BackgammonState, CubeState


class CubeTests(unittest.TestCase):
    def test_centered_cube_can_be_doubled_and_taken(self) -> None:
        state = BackgammonState.initial(turn=0)

        offer = make_offer(state)
        next_state = apply_take(offer)

        self.assertTrue(can_double(state))
        self.assertEqual(offer.offered_value, 2)
        self.assertEqual(next_state.cube.value, 2)
        self.assertEqual(next_state.cube.owner, 1)

    def test_player_cannot_double_when_opponent_owns_cube(self) -> None:
        state = BackgammonState.initial(turn=0)
        state = BackgammonState(
            points=state.points,
            bar=state.bar,
            off=state.off,
            turn=0,
            cube=CubeState(value=2, owner=1),
        )

        self.assertFalse(can_double(state))

    def test_pass_awards_current_cube_value_to_doubler(self) -> None:
        state = BackgammonState.initial(turn=0)
        offer = CubeOffer(state=state, doubler=0, offered_value=2, pass_value=1)

        outcome = apply_pass(offer)

        self.assertEqual(outcome.terminal_winner, 0)
        self.assertEqual(outcome.points_won, 1)

    def test_checker_points_include_gammon_multiplier(self) -> None:
        points = [0] * 24
        points[23] = -2
        points[12] = -5
        points[7] = -3
        points[6] = -5
        state = BackgammonState(points=tuple(points), off=(15, 0), cube=CubeState(value=2))

        self.assertEqual(checker_points_won(state, 0), 4)


if __name__ == "__main__":
    unittest.main()
