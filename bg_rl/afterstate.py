from __future__ import annotations

from dataclasses import dataclass

from bg_rl.state import Action, BackgammonState


@dataclass(frozen=True)
class AfterstateCandidate:
    action: Action
    afterstate: BackgammonState


def legal_afterstates(
    state: BackgammonState, dice: tuple[int, int]
) -> tuple[AfterstateCandidate, ...]:
    return tuple(
        AfterstateCandidate(action=action, afterstate=state.apply_action(action))
        for action in state.legal_actions(dice)
    )


def selected_afterstate(
    state: BackgammonState, action: Action
) -> BackgammonState:
    return state.apply_action(action)
