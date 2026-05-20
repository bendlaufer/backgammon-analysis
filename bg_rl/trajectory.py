from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from bg_rl.mat import CheckerDecision, ParsedMatch
from bg_rl.state import Action, BackgammonState, MoveStep


def state_to_dict(state: BackgammonState) -> dict[str, Any]:
    return {
        "points": list(state.points),
        "bar": list(state.bar),
        "off": list(state.off),
        "turn": state.turn,
        "cube": asdict(state.cube),
        "score": list(state.score),
        "match_length": state.match_length,
    }


def step_to_token(step: MoveStep) -> str:
    source = "bar" if step.from_point is None else str(step.from_point)
    destination = "off" if step.to_point is None else str(step.to_point)
    suffix = "*" if step.hit else ""
    return f"{source}/{destination}:{step.die}{suffix}"


def action_to_tokens(action: Action) -> list[str]:
    return [step_to_token(step) for step in action]


def action_to_key(action: Action) -> str:
    return " ".join(action_to_tokens(action))


def decision_to_record(
    decision: CheckerDecision,
    *,
    source_file: str,
    match_headers: dict[str, str],
    game_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legal_actions = decision.state.legal_actions(decision.dice)
    selected_index = _selected_action_index(decision.action, legal_actions)
    return {
        "source_file": source_file,
        "game_index": decision.game_index,
        "turn_number": decision.turn_number,
        "player": decision.player,
        "dice": list(decision.dice),
        "state": state_to_dict(decision.state),
        "legal_actions": [action_to_tokens(action) for action in legal_actions],
        "legal_action_keys": [action_to_key(action) for action in legal_actions],
        "selected_action": action_to_tokens(decision.action),
        "selected_action_key": action_to_key(decision.action),
        "selected_action_index": selected_index,
        "selected_action_is_full_legal_action": selected_index is not None,
        "raw": decision.raw,
        "event_date": match_headers.get("EventDate", ""),
        "result": match_headers.get("RE", ""),
        "game_outcome": game_outcome,
        "result_from_player_perspective": _result_from_player_perspective(
            decision.player, game_outcome
        ),
    }


def match_to_records(parsed: ParsedMatch, *, source_file: str) -> list[dict[str, Any]]:
    outcomes = {
        outcome.game_index: {
            "winner": outcome.winner,
            "points_won": outcome.points_won,
            "score_after_game": list(outcome.score_after_game),
        }
        for outcome in parsed.game_outcomes
    }
    return [
        decision_to_record(
            decision,
            source_file=source_file,
            match_headers=parsed.headers,
            game_outcome=outcomes.get(decision.game_index),
        )
        for decision in parsed.decisions
    ]


def record_to_json(record: dict[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"), sort_keys=True)


def _selected_action_index(action: Action, legal_actions: tuple[Action, ...]) -> int | None:
    try:
        return legal_actions.index(action)
    except ValueError:
        return None


def _result_from_player_perspective(
    player: int, game_outcome: dict[str, Any] | None
) -> int | None:
    if game_outcome is None:
        return None
    sign = 1 if game_outcome["winner"] == player else -1
    return sign * int(game_outcome["points_won"])
