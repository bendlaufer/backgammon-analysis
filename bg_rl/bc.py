from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset


ACTION_TOKEN_RE = re.compile(r"(?P<src>bar|off|\d+)/(?P<dst>bar|off|\d+):(?P<die>[1-6])(?P<hit>\*)?")
STATE_FEATURE_DIM = 36
DICE_FEATURE_DIM = 12
ACTION_FEATURE_DIM = 24
BC_FEATURE_DIM = STATE_FEATURE_DIM + DICE_FEATURE_DIM + ACTION_FEATURE_DIM


@dataclass(frozen=True)
class BCSample:
    candidate_features: torch.Tensor
    selected_index: int
    source_file: str
    raw: str


class TrajectoryBCDataset(Dataset[BCSample]):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        max_samples: int = 0,
        max_legal_actions: int = 512,
    ) -> None:
        self.samples: list[BCSample] = []
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                if not record.get("selected_action_is_full_legal_action"):
                    continue
                if record["selected_action_index"] is None:
                    continue
                if len(record["legal_actions"]) > max_legal_actions:
                    continue
                self.samples.append(record_to_bc_sample(record))
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BCSample:
        return self.samples[index]


class CandidatePolicyNet(nn.Module):
    def __init__(self, input_dim: int = BC_FEATURE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.net(candidate_features).squeeze(-1)


def record_to_bc_sample(record: dict[str, Any]) -> BCSample:
    state_features = encode_state_record(record["state"])
    dice_features = encode_dice(record["dice"])
    candidate_rows = []
    for action_tokens in record["legal_actions"]:
        candidate_rows.append(
            torch.cat(
                [
                    state_features,
                    dice_features,
                    encode_action_tokens(action_tokens),
                ]
            )
        )
    return BCSample(
        candidate_features=torch.stack(candidate_rows),
        selected_index=int(record["selected_action_index"]),
        source_file=record["source_file"],
        raw=record["raw"],
    )


def collate_bc_samples(samples: list[BCSample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_candidates = max(sample.candidate_features.shape[0] for sample in samples)
    features = torch.zeros(batch_size, max_candidates, BC_FEATURE_DIM, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)

    for row, sample in enumerate(samples):
        count = sample.candidate_features.shape[0]
        features[row, :count] = sample.candidate_features
        mask[row, :count] = True
        labels[row] = sample.selected_index

    return {"features": features, "mask": mask, "labels": labels}


def encode_state_record(state: dict[str, Any]) -> torch.Tensor:
    turn = int(state["turn"])
    points = [float(value) for value in state["points"]]
    if turn == 1:
        points = [-value for value in reversed(points)]

    if turn == 0:
        own_bar, opp_bar = state["bar"]
        own_off, opp_off = state["off"]
        own_score, opp_score = state["score"]
        cube_owner = _cube_owner_feature(state["cube"]["owner"], 0)
    else:
        own_bar, opp_bar = state["bar"][1], state["bar"][0]
        own_off, opp_off = state["off"][1], state["off"][0]
        own_score, opp_score = state["score"][1], state["score"][0]
        cube_owner = _cube_owner_feature(state["cube"]["owner"], 1)

    match_length = state["match_length"] or 1
    features = [value / 15.0 for value in points]
    features.extend(
        [
            own_bar / 15.0,
            opp_bar / 15.0,
            own_off / 15.0,
            opp_off / 15.0,
            1.0,
            state["cube"]["value"] / 64.0,
            cube_owner,
            float(state["cube"]["crawford"]),
            float(state["cube"]["jacoby"]),
            own_score / max(1, match_length),
            opp_score / max(1, match_length),
            float(state["match_length"] or 0) / 25.0,
        ]
    )
    return torch.tensor(features, dtype=torch.float32)


def encode_dice(dice: list[int] | tuple[int, int]) -> torch.Tensor:
    features = torch.zeros(DICE_FEATURE_DIM, dtype=torch.float32)
    for offset, die in enumerate(dice):
        features[offset * 6 + int(die) - 1] = 1.0
    return features


def encode_action_tokens(action_tokens: list[str]) -> torch.Tensor:
    step_features = [encode_step_token(token) for token in action_tokens[:4]]
    while len(step_features) < 4:
        step_features.append(torch.zeros(6, dtype=torch.float32))
    return torch.cat(step_features)


def encode_step_token(token: str) -> torch.Tensor:
    match = ACTION_TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid action token: {token}")
    source = _point_feature(match.group("src"))
    destination = _point_feature(match.group("dst"))
    die = int(match.group("die")) / 6.0
    hit = 1.0 if match.group("hit") else 0.0
    is_bar = 1.0 if match.group("src") == "bar" else 0.0
    is_off = 1.0 if match.group("dst") == "off" else 0.0
    return torch.tensor([source, destination, die, hit, is_bar, is_off], dtype=torch.float32)


def masked_cross_entropy(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    return nn.functional.cross_entropy(masked_logits, labels)


def masked_accuracy(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> float:
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    predictions = masked_logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def _point_feature(point: str) -> float:
    if point == "bar":
        return 1.0
    if point == "off":
        return 0.0
    return (int(point) + 1) / 24.0


def _cube_owner_feature(owner: int | None, perspective: int) -> float:
    if owner is None or (isinstance(owner, float) and math.isnan(owner)):
        return 0.0
    return 1.0 if int(owner) == perspective else -1.0
