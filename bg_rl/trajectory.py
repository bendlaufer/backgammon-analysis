from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from bg_rl.mat import CheckerDecision, ParsedMatch
from bg_rl.state import Action, BackgammonState, CubeState, MoveStep


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


def action_from_tokens(tokens: list[str]) -> Action:
    steps: list[MoveStep] = []
    for token in tokens:
        move, die_text = token.split(":", maxsplit=1)
        source_text, destination_text = move.split("/", maxsplit=1)
        hit = die_text.endswith("*")
        die = int(die_text.rstrip("*"))
        source = None if source_text == "bar" else int(source_text)
        destination = None if destination_text == "off" else int(destination_text)
        steps.append(MoveStep(source, destination, die, hit))
    return tuple(steps)


def state_from_dict(record: dict[str, Any]) -> BackgammonState:
    cube = record.get("cube", {})
    return BackgammonState(
        points=tuple(record["points"]),
        bar=tuple(record["bar"]),
        off=tuple(record["off"]),
        turn=int(record["turn"]),
        cube=CubeState(
            value=int(cube.get("value", 1)),
            owner=cube.get("owner"),
            crawford=bool(cube.get("crawford", False)),
            jacoby=bool(cube.get("jacoby", False)),
        ),
        score=tuple(record["score"]),
        match_length=record.get("match_length"),
    )


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


def decision_to_compact_record(
    decision: CheckerDecision,
    *,
    source_file: str,
    match_headers: dict[str, str],
    game_outcome: dict[str, Any] | None = None,
    trust_parser_validation: bool = False,
) -> dict[str, Any]:
    if trust_parser_validation:
        is_full_legal_action = True
    else:
        legal_actions = decision.state.legal_actions(decision.dice)
        is_full_legal_action = _selected_action_index(decision.action, legal_actions) is not None
    return {
        "format": "compact_checker_decision_v1",
        "source_file": source_file,
        "game_index": decision.game_index,
        "turn_number": decision.turn_number,
        "player": decision.player,
        "dice": list(decision.dice),
        "state": state_to_dict(decision.state),
        "selected_action": action_to_tokens(decision.action),
        "selected_action_key": action_to_key(decision.action),
        "selected_action_is_full_legal_action": is_full_legal_action,
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


def match_to_compact_records(
    parsed: ParsedMatch,
    *,
    source_file: str,
    trust_parser_validation: bool = False,
) -> list[dict[str, Any]]:
    outcomes = {
        outcome.game_index: {
            "winner": outcome.winner,
            "points_won": outcome.points_won,
            "score_after_game": list(outcome.score_after_game),
        }
        for outcome in parsed.game_outcomes
    }
    return [
        decision_to_compact_record(
            decision,
            source_file=source_file,
            match_headers=parsed.headers,
            game_outcome=outcomes.get(decision.game_index),
            trust_parser_validation=trust_parser_validation,
        )
        for decision in parsed.decisions
    ]


def legal_actions_from_record(record: dict[str, Any]) -> tuple[Action, ...]:
    return state_from_dict(record["state"]).legal_actions(tuple(record["dice"]))


def selected_action_index_from_record(record: dict[str, Any]) -> int | None:
    action = action_from_tokens(record["selected_action"])
    return _selected_action_index(action, legal_actions_from_record(record))


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
