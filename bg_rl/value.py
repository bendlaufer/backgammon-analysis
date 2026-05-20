from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from bg_rl.bc import STATE_FEATURE_DIM, encode_state_record


class TrajectoryValueDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        max_samples: int = 0,
        target_scale: float = 16.0,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                target = value_target_from_record(record, target_scale=target_scale)
                if target is None:
                    continue
                self.samples.append((encode_state_record(record["state"]), torch.tensor(target)))
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


class ValueNet(nn.Module):
    def __init__(self, input_dim: int = STATE_FEATURE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features).squeeze(-1)


def value_target_from_record(record: dict[str, Any], *, target_scale: float = 16.0) -> float | None:
    result = record.get("result_from_player_perspective")
    if result is None:
        return None
    return max(-1.0, min(1.0, float(result) / target_scale))
