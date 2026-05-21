from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from bg_rl.bc import CandidatePolicyNet, record_to_bc_sample
from bg_rl.state import Action, BackgammonState
from bg_rl.trajectory import action_to_key, action_to_tokens, state_to_dict


@dataclass(frozen=True)
class MoveRanking:
    action: Action
    score: float


class BasicEngine:
    """A simple legal-move engine.

    If a behavior-cloning checkpoint is available, actions are scored by the
    trained candidate scorer. Otherwise, a deterministic heuristic is used.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model: CandidatePolicyNet | None = None
        if model_path is not None and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            hidden_dim = int(checkpoint.get("hidden_dim", 128))
            model = CandidatePolicyNet(hidden_dim=hidden_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model

    def rank_moves(
        self, state: BackgammonState, dice: tuple[int, int]
    ) -> tuple[MoveRanking, ...]:
        legal_actions = state.legal_actions(dice)
        if not legal_actions:
            return ()
        if self.model is not None:
            return self._rank_with_model(state, dice, legal_actions)
        return tuple(
            sorted(
                (MoveRanking(action=action, score=_heuristic_score(state, action)) for action in legal_actions),
                key=lambda ranking: ranking.score,
                reverse=True,
            )
        )

    def choose_move(self, state: BackgammonState, dice: tuple[int, int]) -> Action:
        rankings = self.rank_moves(state, dice)
        return rankings[0].action if rankings else ()

    def _rank_with_model(
        self,
        state: BackgammonState,
        dice: tuple[int, int],
        legal_actions: tuple[Action, ...],
    ) -> tuple[MoveRanking, ...]:
        assert self.model is not None
        record = {
            "state": state_to_dict(state),
            "dice": list(dice),
            "legal_actions": [action_to_tokens(action) for action in legal_actions],
            "selected_action_index": 0,
            "selected_action_is_full_legal_action": True,
            "source_file": "",
            "raw": "",
        }
        sample = record_to_bc_sample(record)
        with torch.no_grad():
            scores = self.model(sample.candidate_features).tolist()
        return tuple(
            sorted(
                (
                    MoveRanking(action=action, score=float(score))
                    for action, score in zip(legal_actions, scores)
                ),
                key=lambda ranking: ranking.score,
                reverse=True,
            )
        )


def engine_summary(engine: BasicEngine) -> str:
    return "bc-policy" if engine.model is not None else "heuristic"


def rankings_to_records(rankings: tuple[MoveRanking, ...]) -> list[dict[str, object]]:
    return [
        {
            "action": action_to_tokens(ranking.action),
            "action_key": action_to_key(ranking.action),
            "score": ranking.score,
        }
        for ranking in rankings
    ]


def _heuristic_score(state: BackgammonState, action: Action) -> float:
    afterstate = state.apply_action(action)
    player = state.turn
    opponent = 1 - player
    pip_before = _pip_count(state, player)
    pip_after = _pip_count(afterstate, player)
    opponent_bar_gain = afterstate.bar[opponent] - state.bar[opponent]
    off_gain = afterstate.off[player] - state.off[player]
    made_points = _made_points(afterstate, player)
    blots = _blot_count(afterstate, player)
    return (
        (pip_before - pip_after)
        + 18.0 * opponent_bar_gain
        + 24.0 * off_gain
        + 1.5 * made_points
        - 1.0 * blots
    )


def _pip_count(state: BackgammonState, player: int) -> int:
    if player == 0:
        board_pips = sum((point + 1) * max(0, count) for point, count in enumerate(state.points))
    else:
        board_pips = sum((24 - point) * max(0, -count) for point, count in enumerate(state.points))
    return board_pips + 25 * state.bar[player]


def _made_points(state: BackgammonState, player: int) -> int:
    if player == 0:
        return sum(1 for count in state.points if count >= 2)
    return sum(1 for count in state.points if count <= -2)


def _blot_count(state: BackgammonState, player: int) -> int:
    if player == 0:
        return sum(1 for count in state.points if count == 1)
    return sum(1 for count in state.points if count == -1)
