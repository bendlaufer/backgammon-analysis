from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bg_rl.bc import (
    BC_FEATURE_DIM,
    STATE_FEATURE_DIM,
    encode_action_tokens,
    encode_dice,
    encode_state_record,
)
from bg_rl.encoding import encode_state
from bg_rl.state import Action, BackgammonState
from bg_rl.trajectory import action_to_tokens, state_to_dict


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 1
    temperature: float = 1.0
    exploration: float = 1.25


@dataclass(frozen=True)
class SearchResult:
    legal_actions: tuple[Action, ...]
    action_probabilities: tuple[float, ...]
    selected_action: Action
    root_value: float


class PolicyValueNet(nn.Module):
    """Dual-head network for AlphaZero-style checker decisions.

    The policy head scores state/dice/action candidates, which avoids a fixed
    action vocabulary for backgammon's variable legal-action set. The value
    head estimates normalized equity from the current player's perspective.
    """

    def __init__(
        self,
        *,
        candidate_input_dim: int = BC_FEATURE_DIM,
        state_input_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def policy_logits(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.policy(candidate_features).squeeze(-1)

    def value_estimate(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.value(state_features).squeeze(-1)


def load_policy_value_model(model_path: str | Path | None, *, hidden_dim: int = 256) -> PolicyValueNet:
    model = PolicyValueNet(hidden_dim=hidden_dim)
    if model_path is not None and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint_hidden_dim = int(checkpoint.get("hidden_dim", hidden_dim))
        model = PolicyValueNet(hidden_dim=checkpoint_hidden_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class SearchMovePolicy:
    """Move policy that records search targets for later policy/value training."""

    def __init__(
        self,
        model: PolicyValueNet,
        *,
        rng: random.Random | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.model = model
        self.rng = rng or random.Random()
        self.config = config or SearchConfig()
        self.history: list[SearchResult] = []

    def choose_move(self, state: BackgammonState, dice: tuple[int, int]) -> Action:
        result = search_checker_move(state, dice, self.model, rng=self.rng, config=self.config)
        self.history.append(result)
        return result.selected_action


def search_checker_move(
    state: BackgammonState,
    dice: tuple[int, int],
    model: PolicyValueNet,
    *,
    rng: random.Random | None = None,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Return a PUCT-compatible search result for one checker decision.

    This first version is a batched one-ply policy/value search: it produces
    the same replay targets as a deeper tree search, while keeping the search
    implementation replaceable when we add chance-node expansion and rollouts.
    """

    rng = rng or random.Random()
    config = config or SearchConfig()
    legal_actions = state.legal_actions(dice)
    if not legal_actions:
        return SearchResult((), (), (), root_value=_value(model, state))

    features = candidate_features(state, dice, legal_actions)
    with torch.no_grad():
        logits = model.policy_logits(features)
    probabilities = _softmax(logits.tolist(), temperature=config.temperature)
    selected_index = _sample_index(probabilities, rng)
    return SearchResult(
        legal_actions=legal_actions,
        action_probabilities=tuple(probabilities),
        selected_action=legal_actions[selected_index],
        root_value=_value(model, state),
    )


def candidate_features(
    state: BackgammonState,
    dice: tuple[int, int],
    legal_actions: tuple[Action, ...],
) -> torch.Tensor:
    record_state = state_to_dict(state)
    state_features = encode_state_record(record_state)
    dice_features = encode_dice(dice)
    rows = [
        torch.cat([state_features, dice_features, encode_action_tokens(action_to_tokens(action))])
        for action in legal_actions
    ]
    return torch.stack(rows)


def rl_record_to_json(
    *,
    game_index: int,
    ply: int,
    player: int,
    state: BackgammonState,
    dice: tuple[int, int],
    result: SearchResult,
    reward: float,
    points_reward: int,
) -> str:
    record = {
        "format": "rl_checker_decision_v1",
        "game_index": game_index,
        "ply": ply,
        "player": player,
        "state": state_to_dict(state),
        "dice": list(dice),
        "legal_actions": [action_to_tokens(action) for action in result.legal_actions],
        "selected_action": action_to_tokens(result.selected_action),
        "search_policy": list(result.action_probabilities),
        "root_value": result.root_value,
        "reward": reward,
        "points_reward": points_reward,
    }
    return json.dumps(record, separators=(",", ":"), sort_keys=True)


def parse_rl_record(line: str) -> dict[str, Any]:
    record = json.loads(line)
    if record.get("format") != "rl_checker_decision_v1":
        raise ValueError(f"unsupported RL replay format: {record.get('format')}")
    return record


def state_value_features(state: BackgammonState) -> torch.Tensor:
    return encode_state(state, perspective=state.turn)


def _value(model: PolicyValueNet, state: BackgammonState) -> float:
    with torch.no_grad():
        return float(model.value_estimate(state_value_features(state).unsqueeze(0)).item())


def _softmax(values: list[float], *, temperature: float) -> list[float]:
    if temperature <= 0:
        best = max(range(len(values)), key=values.__getitem__)
        return [1.0 if index == best else 0.0 for index in range(len(values))]
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _sample_index(probabilities: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    total = 0.0
    for index, probability in enumerate(probabilities):
        total += probability
        if threshold <= total:
            return index
    return len(probabilities) - 1
