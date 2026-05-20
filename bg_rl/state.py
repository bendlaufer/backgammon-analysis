from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Player = int
Point = int | None


@dataclass(frozen=True)
class CubeState:
    """Doubling-cube state carried by the environment.

    owner is None for centered, 0 for player 0 ownership, and 1 for player 1.
    The checker-play simulator does not yet generate cube actions, but value
    training needs the cube context present from the start.
    """

    value: int = 1
    owner: Player | None = None
    crawford: bool = False
    jacoby: bool = False


@dataclass(frozen=True)
class MoveStep:
    """One checker movement using one die.

    from_point/to_point are 0-based board points. None means bar for
    from_point and borne off for to_point.
    """

    from_point: Point
    to_point: Point
    die: int
    hit: bool = False


Action = tuple[MoveStep, ...]


@dataclass(frozen=True)
class BackgammonState:
    """Immutable backgammon checker state.

    points has 24 entries. Positive values are player 0 checkers, negative
    values are player 1 checkers. Player 0 moves from point 23 down to 0;
    player 1 moves from point 0 up to 23.
    """

    points: tuple[int, ...]
    bar: tuple[int, int] = (0, 0)
    off: tuple[int, int] = (0, 0)
    turn: Player = 0
    cube: CubeState = CubeState()
    score: tuple[int, int] = (0, 0)
    match_length: int | None = None

    def __post_init__(self) -> None:
        if len(self.points) != 24:
            raise ValueError("points must contain exactly 24 entries")
        if self.turn not in (0, 1):
            raise ValueError("turn must be 0 or 1")
        for name, pair in (("bar", self.bar), ("off", self.off), ("score", self.score)):
            if len(pair) != 2:
                raise ValueError(f"{name} must have two entries")
        if any(value < 0 for value in self.bar + self.off + self.score):
            raise ValueError("bar, off, and score counts must be non-negative")
        self._validate_checker_counts()

    @classmethod
    def initial(cls, *, turn: Player = 0) -> BackgammonState:
        points = [0] * 24
        points[23] = 2
        points[12] = 5
        points[7] = 3
        points[5] = 5
        points[0] = -2
        points[11] = -5
        points[16] = -3
        points[18] = -5
        return cls(points=tuple(points), turn=turn)

    def legal_actions(self, dice: tuple[int, int]) -> tuple[Action, ...]:
        from bg_rl.rules import legal_checker_actions

        return legal_checker_actions(self, dice)

    def apply_action(self, action: Iterable[MoveStep]) -> BackgammonState:
        from bg_rl.rules import apply_action

        return apply_action(self, tuple(action))

    def current_player_won(self) -> bool:
        return self.off[self.turn] == 15

    def winner(self) -> Player | None:
        if self.off[0] == 15:
            return 0
        if self.off[1] == 15:
            return 1
        return None

    def point_count(self, player: Player) -> int:
        if player == 0:
            return sum(max(0, value) for value in self.points)
        return sum(max(0, -value) for value in self.points)

    def total_checkers(self, player: Player) -> int:
        return self.point_count(player) + self.bar[player] + self.off[player]

    def _validate_checker_counts(self) -> None:
        for player in (0, 1):
            total = self.total_checkers(player)
            if total != 15:
                raise ValueError(f"player {player} has {total} checkers, expected 15")

    def swap_turn(self) -> BackgammonState:
        return BackgammonState(
            points=self.points,
            bar=self.bar,
            off=self.off,
            turn=1 - self.turn,
            cube=self.cube,
            score=self.score,
            match_length=self.match_length,
        )
