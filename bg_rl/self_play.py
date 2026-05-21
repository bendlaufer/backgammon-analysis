from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Protocol

from bg_rl.cube import (
    CubePolicy,
    DoubleDecision,
    HeuristicCubePolicy,
    NoDoubleTakePolicy,
    TakeDecision,
    apply_pass,
    apply_take,
    can_double,
    checker_points_won,
    make_offer,
)
from bg_rl.engine import BasicEngine
from bg_rl.state import Action, BackgammonState
from bg_rl.trajectory import action_to_key, action_to_tokens, state_to_dict


class MovePolicy(Protocol):
    def choose_move(self, state: BackgammonState, dice: tuple[int, int]) -> Action:
        ...


@dataclass(frozen=True)
class SelfPlayDecision:
    game_index: int
    ply: int
    phase: str
    player: int
    state: BackgammonState
    dice: tuple[int, int] | None = None
    action: Action | None = None
    double_decision: DoubleDecision | None = None
    take_decision: TakeDecision | None = None
    legal_action_count: int | None = None
    reward: float | None = None
    points_reward: int | None = None


@dataclass(frozen=True)
class SelfPlayGame:
    game_index: int
    decisions: tuple[SelfPlayDecision, ...]
    winner: int | None
    points_won: int
    plies: int


class RandomPolicy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def choose_move(self, state: BackgammonState, dice: tuple[int, int]) -> Action:
        legal_actions = state.legal_actions(dice)
        return self.rng.choice(legal_actions) if legal_actions else ()


def roll_dice(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)


def play_game(
    *,
    game_index: int = 0,
    policy0: MovePolicy | None = None,
    policy1: MovePolicy | None = None,
    cube_policy0: CubePolicy | None = None,
    cube_policy1: CubePolicy | None = None,
    rng: random.Random | None = None,
    max_plies: int = 2000,
) -> SelfPlayGame:
    rng = rng or random.Random()
    move_policies: tuple[MovePolicy, MovePolicy] = (
        policy0 or BasicEngine(),
        policy1 or BasicEngine(),
    )
    cube_policies: tuple[CubePolicy, CubePolicy] = (
        cube_policy0 or NoDoubleTakePolicy(),
        cube_policy1 or NoDoubleTakePolicy(),
    )
    state = BackgammonState.initial(turn=0)
    decisions: list[SelfPlayDecision] = []

    for ply in range(max_plies):
        checker_winner = state.winner()
        if checker_winner is not None:
            return _with_rewards(
                game_index,
                decisions,
                winner=checker_winner,
                points_won=checker_points_won(state, checker_winner),
            )

        if can_double(state):
            double_decision = cube_policies[state.turn].choose_double(state)
            decisions.append(
                SelfPlayDecision(
                    game_index=game_index,
                    ply=ply,
                    phase="double",
                    player=state.turn,
                    state=state,
                    double_decision=double_decision,
                )
            )
            if double_decision == DoubleDecision.DOUBLE:
                offer = make_offer(state)
                taker = 1 - state.turn
                take_decision = cube_policies[taker].choose_take(offer)
                decisions.append(
                    SelfPlayDecision(
                        game_index=game_index,
                        ply=ply,
                        phase="take",
                        player=taker,
                        state=state,
                        take_decision=take_decision,
                    )
                )
                if take_decision == TakeDecision.PASS:
                    outcome = apply_pass(offer)
                    return _with_rewards(
                        game_index,
                        decisions,
                        winner=outcome.terminal_winner,
                        points_won=outcome.points_won,
                    )
                state = apply_take(offer)

        dice = roll_dice(rng)
        legal_actions = state.legal_actions(dice)
        action = move_policies[state.turn].choose_move(state, dice)
        if action not in legal_actions:
            raise ValueError(f"policy selected illegal action: {action}")
        decisions.append(
            SelfPlayDecision(
                game_index=game_index,
                ply=ply,
                phase="checker",
                player=state.turn,
                dice=dice,
                state=state,
                action=action,
                legal_action_count=len(legal_actions),
            )
        )
        state = state.apply_action(action)

    return _with_rewards(game_index, decisions, winner=None, points_won=0)


def self_play_record_to_json(decision: SelfPlayDecision) -> str:
    record = {
        "format": "self_play_decision_v2",
        "game_index": decision.game_index,
        "ply": decision.ply,
        "phase": decision.phase,
        "player": decision.player,
        "state": state_to_dict(decision.state),
        "reward": decision.reward,
        "points_reward": decision.points_reward,
    }
    if decision.phase == "checker":
        record.update(
            {
                "dice": list(decision.dice or ()),
                "selected_action": action_to_tokens(decision.action or ()),
                "selected_action_key": action_to_key(decision.action or ()),
                "legal_action_count": decision.legal_action_count,
            }
        )
    elif decision.phase == "double":
        record["double_decision"] = decision.double_decision.value if decision.double_decision else None
    elif decision.phase == "take":
        record["take_decision"] = decision.take_decision.value if decision.take_decision else None
    return json.dumps(record, separators=(",", ":"), sort_keys=True)


def _with_rewards(
    game_index: int,
    decisions: list[SelfPlayDecision],
    *,
    winner: int | None,
    points_won: int,
) -> SelfPlayGame:
    rewarded = tuple(
        SelfPlayDecision(
            game_index=decision.game_index,
            ply=decision.ply,
            phase=decision.phase,
            player=decision.player,
            state=decision.state,
            dice=decision.dice,
            action=decision.action,
            double_decision=decision.double_decision,
            take_decision=decision.take_decision,
            legal_action_count=decision.legal_action_count,
            reward=_reward(decision.player, winner),
            points_reward=_points_reward(decision.player, winner, points_won),
        )
        for decision in decisions
    )
    return SelfPlayGame(
        game_index=game_index,
        decisions=rewarded,
        winner=winner,
        points_won=points_won,
        plies=sum(1 for decision in decisions if decision.phase == "checker"),
    )


def build_cube_policy(name: str) -> CubePolicy:
    if name == "heuristic":
        return HeuristicCubePolicy()
    if name == "none":
        return NoDoubleTakePolicy()
    raise ValueError(f"unknown cube policy: {name}")


def _reward(player: int, winner: int | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if player == winner else -1.0


def _points_reward(player: int, winner: int | None, points_won: int) -> int:
    if winner is None:
        return 0
    return points_won if player == winner else -points_won
