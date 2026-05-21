from __future__ import annotations

from dataclasses import dataclass
import json
from statistics import mean

from bg_rl.trajectory import legal_actions_from_record, selected_action_index_from_record


@dataclass(frozen=True)
class TrajectoryStats:
    rows: int
    full_legal_labels: int
    trainable_rows: int
    candidate_counts: tuple[int, ...]

    @property
    def partial_or_unindexed_labels(self) -> int:
        return self.rows - self.full_legal_labels

    @property
    def mean_legal_actions(self) -> float:
        return mean(self.candidate_counts) if self.candidate_counts else 0.0

    def percentile(self, percentile: int) -> int:
        if not self.candidate_counts:
            return 0
        sorted_counts = sorted(self.candidate_counts)
        index = int((len(sorted_counts) - 1) * percentile / 100)
        return sorted_counts[index]


def summarize_jsonl_lines(
    lines: list[str],
    *,
    max_legal_actions: int,
    recompute_compact_legal_actions: bool = True,
) -> TrajectoryStats:
    rows = 0
    full = 0
    trainable = 0
    candidate_counts: list[int] = []

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        rows += 1
        if "legal_actions" in record:
            candidate_count = len(record["legal_actions"])
            selected_index = record.get("selected_action_index")
        elif recompute_compact_legal_actions:
            candidate_count = len(legal_actions_from_record(record))
            selected_index = selected_action_index_from_record(record)
        else:
            candidate_count = -1
            selected_index = 0 if record.get("selected_action_is_full_legal_action") else None

        if candidate_count >= 0:
            candidate_counts.append(candidate_count)
        if selected_index is not None:
            full += 1
            if 0 <= candidate_count <= max_legal_actions:
                trainable += 1

    return TrajectoryStats(
        rows=rows,
        full_legal_labels=full,
        trainable_rows=trainable,
        candidate_counts=tuple(candidate_counts),
    )


def merge_stats(stats: list[TrajectoryStats]) -> TrajectoryStats:
    return TrajectoryStats(
        rows=sum(stat.rows for stat in stats),
        full_legal_labels=sum(stat.full_legal_labels for stat in stats),
        trainable_rows=sum(stat.trainable_rows for stat in stats),
        candidate_counts=tuple(
            count for stat in stats for count in stat.candidate_counts
        ),
    )
