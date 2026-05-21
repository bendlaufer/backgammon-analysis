from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from bg_rl.engine import BasicEngine
from bg_rl.state import Action, BackgammonState
from bg_rl.trajectory import action_from_tokens, action_to_key, action_to_tokens, state_from_dict


@dataclass(frozen=True)
class BenchmarkPosition:
    id: str
    title: str
    category: str
    status: str
    record: dict[str, Any]

    @property
    def is_checker_decision(self) -> bool:
        return self.record.get("decision", {}).get("kind") == "checker"

    @property
    def state(self) -> BackgammonState | None:
        state_record = self.record.get("position", {}).get("state")
        if state_record is None:
            return None
        return state_from_dict(state_record)

    @property
    def dice(self) -> tuple[int, int] | None:
        dice = self.record.get("decision", {}).get("dice")
        if dice is None:
            return None
        return int(dice[0]), int(dice[1])

    @property
    def expert_actions(self) -> tuple[Action, ...]:
        actions = []
        for tokens in self.record.get("decision", {}).get("expert_actions", []):
            actions.append(action_from_tokens(tokens))
        return tuple(actions)


def load_benchmarks(path: str | Path) -> list[BenchmarkPosition]:
    benchmarks: list[BenchmarkPosition] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            record = json.loads(stripped)
            benchmark = BenchmarkPosition(
                id=_required_str(record, "id", line_number),
                title=_required_str(record, "title", line_number),
                category=_required_str(record, "category", line_number),
                status=_required_str(record, "status", line_number),
                record=record,
            )
            validate_benchmark(benchmark)
            benchmarks.append(benchmark)
    return benchmarks


def validate_benchmark(benchmark: BenchmarkPosition) -> None:
    if benchmark.status not in {"active", "candidate", "opaque"}:
        raise ValueError(f"{benchmark.id}: invalid status {benchmark.status!r}")
    if "source" not in benchmark.record:
        raise ValueError(f"{benchmark.id}: missing source")
    if "position" not in benchmark.record:
        raise ValueError(f"{benchmark.id}: missing position")
    if "decision" not in benchmark.record:
        raise ValueError(f"{benchmark.id}: missing decision")
    if benchmark.is_checker_decision and benchmark.state is not None:
        dice = benchmark.dice
        if dice is None:
            raise ValueError(f"{benchmark.id}: checker decision with state needs dice")
        legal_actions = benchmark.state.legal_actions(dice)
        for action in benchmark.expert_actions:
            if action not in legal_actions:
                raise ValueError(
                    f"{benchmark.id}: expert action {action_to_key(action)!r} is not legal"
                )


def score_benchmark(
    benchmark: BenchmarkPosition,
    engine: BasicEngine,
) -> dict[str, Any]:
    state = benchmark.state
    dice = benchmark.dice
    if state is None or dice is None or not benchmark.is_checker_decision:
        return {
            "id": benchmark.id,
            "status": benchmark.status,
            "scoreable": False,
            "reason": "missing parsed state/dice or non-checker decision",
        }
    rankings = engine.rank_moves(state, dice)
    expert_keys = {action_to_key(action) for action in benchmark.expert_actions}
    top_key = action_to_key(rankings[0].action) if rankings else ""
    best_expert_rank = None
    for index, ranking in enumerate(rankings, start=1):
        if action_to_key(ranking.action) in expert_keys:
            best_expert_rank = index
            break
    return {
        "id": benchmark.id,
        "status": benchmark.status,
        "scoreable": True,
        "engine_top_action": action_to_tokens(rankings[0].action) if rankings else [],
        "engine_top_key": top_key,
        "engine_top_score": rankings[0].score if rankings else None,
        "expert_action_keys": sorted(expert_keys),
        "best_expert_rank": best_expert_rank,
        "legal_action_count": len(rankings),
        "engine_matches_expert": top_key in expert_keys,
    }


def _required_str(record: dict[str, Any], key: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"line {line_number}: missing string field {key!r}")
    return value
