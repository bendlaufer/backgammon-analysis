from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from bg_rl.state import BackgammonState, CubeState


class DoubleDecision(str, Enum):
    NO_DOUBLE = "no_double"
    DOUBLE = "double"


class TakeDecision(str, Enum):
    TAKE = "take"
    PASS = "pass"


@dataclass(frozen=True)
class CubeOffer:
    state: BackgammonState
    doubler: int
    offered_value: int
    pass_value: int


@dataclass(frozen=True)
class CubeOutcome:
    state: BackgammonState
    terminal_winner: int | None = None
    points_won: int = 0


class CubePolicy(Protocol):
    def choose_double(self, state: BackgammonState) -> DoubleDecision:
        ...

    def choose_take(self, offer: CubeOffer) -> TakeDecision:
        ...


class NoDoubleTakePolicy:
    def choose_double(self, state: BackgammonState) -> DoubleDecision:
        return DoubleDecision.NO_DOUBLE

    def choose_take(self, offer: CubeOffer) -> TakeDecision:
        return TakeDecision.TAKE


class HeuristicCubePolicy:
    """Small deterministic cube policy for environment smoke tests.

    This is not intended to be strong. It exists so the self-play environment
    exercises double/take/pass phases before learned cube policies are added.
    """

    def choose_double(self, state: BackgammonState) -> DoubleDecision:
        if not can_double(state):
            return DoubleDecision.NO_DOUBLE
        pip_diff = _pip_count(state, 1 - state.turn) - _pip_count(state, state.turn)
        return DoubleDecision.DOUBLE if pip_diff >= 20 else DoubleDecision.NO_DOUBLE

    def choose_take(self, offer: CubeOffer) -> TakeDecision:
        taker = 1 - offer.doubler
        pip_diff = _pip_count(offer.state, offer.doubler) - _pip_count(offer.state, taker)
        return TakeDecision.PASS if pip_diff >= 35 else TakeDecision.TAKE


def can_double(state: BackgammonState) -> bool:
    if state.cube.crawford:
        return False
    return state.cube.owner is None or state.cube.owner == state.turn


def make_offer(state: BackgammonState) -> CubeOffer:
    if not can_double(state):
        raise ValueError("player cannot double from this cube state")
    return CubeOffer(
        state=state,
        doubler=state.turn,
        offered_value=state.cube.value * 2,
        pass_value=state.cube.value,
    )


def apply_take(offer: CubeOffer) -> BackgammonState:
    taker = 1 - offer.doubler
    return BackgammonState(
        points=offer.state.points,
        bar=offer.state.bar,
        off=offer.state.off,
        turn=offer.state.turn,
        cube=CubeState(
            value=offer.offered_value,
            owner=taker,
            crawford=offer.state.cube.crawford,
            jacoby=offer.state.cube.jacoby,
        ),
        score=offer.state.score,
        match_length=offer.state.match_length,
    )


def apply_pass(offer: CubeOffer) -> CubeOutcome:
    return CubeOutcome(
        state=offer.state,
        terminal_winner=offer.doubler,
        points_won=offer.pass_value,
    )


def checker_points_won(state: BackgammonState, winner: int) -> int:
    loser = 1 - winner
    multiplier = 1
    if state.off[loser] == 0:
        multiplier = 2
        if _backgammon_loss(state, winner, loser):
            multiplier = 3
    return state.cube.value * multiplier


def _backgammon_loss(state: BackgammonState, winner: int, loser: int) -> bool:
    if state.bar[loser] > 0:
        return True
    if winner == 0:
        return any(count < 0 for count in state.points[:6])
    return any(count > 0 for count in state.points[18:])


def _pip_count(state: BackgammonState, player: int) -> int:
    if player == 0:
        board_pips = sum((point + 1) * max(0, count) for point, count in enumerate(state.points))
    else:
        board_pips = sum((24 - point) * max(0, -count) for point, count in enumerate(state.points))
    return board_pips + 25 * state.bar[player]
