from __future__ import annotations

import torch

from bg_rl.state import BackgammonState


def encode_state(state: BackgammonState, *, perspective: int | None = None) -> torch.Tensor:
    """Encode a state as a compact float tensor for policy/value networks.

    Shape is (36,): 24 signed point occupancies from the chosen perspective,
    bar/off counts for both players, side-to-move, cube value/owner, score, and
    optional match length. This is deliberately simple; richer feature planes
    can be added after the simulator and tests stabilize.
    """

    if perspective is None:
        perspective = state.turn
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")

    if perspective == 0:
        points = list(state.points)
        own_bar, opp_bar = state.bar
        own_off, opp_off = state.off
        own_score, opp_score = state.score
        turn_feature = 1.0 if state.turn == 0 else -1.0
        cube_owner = _cube_owner_feature(state.cube.owner, 0)
    else:
        points = [-value for value in reversed(state.points)]
        own_bar, opp_bar = state.bar[1], state.bar[0]
        own_off, opp_off = state.off[1], state.off[0]
        own_score, opp_score = state.score[1], state.score[0]
        turn_feature = 1.0 if state.turn == 1 else -1.0
        cube_owner = _cube_owner_feature(state.cube.owner, 1)

    features = [value / 15.0 for value in points]
    features.extend(
        [
            own_bar / 15.0,
            opp_bar / 15.0,
            own_off / 15.0,
            opp_off / 15.0,
            turn_feature,
            state.cube.value / 64.0,
            cube_owner,
            float(state.cube.crawford),
            float(state.cube.jacoby),
            own_score / max(1, state.match_length or 1),
            opp_score / max(1, state.match_length or 1),
            float(state.match_length or 0) / 25.0,
        ]
    )
    return torch.tensor(features, dtype=torch.float32)


def _cube_owner_feature(owner: int | None, perspective: int) -> float:
    if owner is None:
        return 0.0
    return 1.0 if owner == perspective else -1.0
