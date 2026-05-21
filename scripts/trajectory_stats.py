import argparse
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.stats import TrajectoryStats, merge_stats, summarize_jsonl_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exported trajectory JSONL rows.")
    parser.add_argument("--data", default="artifacts/trajectories/checker_decisions.jsonl")
    parser.add_argument("--max-legal-actions", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument(
        "--skip-compact-legal-actions",
        action="store_true",
        help="For compact rows, skip legal-action regeneration and report only label counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = summarize_path(
        Path(args.data),
        max_legal_actions=args.max_legal_actions,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        recompute_compact_legal_actions=not args.skip_compact_legal_actions,
    )
    print_stats(stats, args.max_legal_actions)


def summarize_path(
    path: Path,
    *,
    max_legal_actions: int,
    num_workers: int,
    chunk_size: int,
    max_rows: int,
    progress_every: int,
    recompute_compact_legal_actions: bool,
) -> TrajectoryStats:
    chunks = iter_line_chunks(path, chunk_size=chunk_size, max_rows=max_rows)
    if num_workers <= 1:
        partials = []
        rows_seen = 0
        next_progress = progress_every
        for chunk in chunks:
            partial = summarize_jsonl_lines(
                chunk,
                max_legal_actions=max_legal_actions,
                recompute_compact_legal_actions=recompute_compact_legal_actions,
            )
            partials.append(partial)
            rows_seen += partial.rows
            if progress_every > 0 and rows_seen >= next_progress:
                print(f"processed_rows={rows_seen}")
                next_progress += progress_every
        return merge_stats(partials)

    tasks = (
        (chunk, max_legal_actions, recompute_compact_legal_actions)
        for chunk in chunks
    )
    partials = []
    rows_seen = 0
    next_progress = progress_every
    with Pool(processes=num_workers) as pool:
        for partial in pool.imap_unordered(_summarize_chunk, tasks):
            partials.append(partial)
            rows_seen += partial.rows
            if progress_every > 0 and rows_seen >= next_progress:
                print(f"processed_rows={rows_seen}")
                next_progress += progress_every
    return merge_stats(partials)


def iter_line_chunks(path: Path, *, chunk_size: int, max_rows: int) -> Iterator[list[str]]:
    current: list[str] = []
    rows = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            current.append(line)
            rows += 1
            if len(current) >= chunk_size:
                yield current
                current = []
            if max_rows > 0 and rows >= max_rows:
                break
    if current:
        yield current


def _summarize_chunk(args: tuple[list[str], int, bool]) -> TrajectoryStats:
    lines, max_legal_actions, recompute = args
    return summarize_jsonl_lines(
        lines,
        max_legal_actions=max_legal_actions,
        recompute_compact_legal_actions=recompute,
    )


def print_stats(stats: TrajectoryStats, max_legal_actions: int) -> None:
    print(f"rows={stats.rows}")
    print(f"full_legal_labels={stats.full_legal_labels}")
    print(f"partial_or_unindexed_labels={stats.partial_or_unindexed_labels}")
    print(f"trainable_at_max_legal_actions_{max_legal_actions}={stats.trainable_rows}")
    if stats.candidate_counts:
        print(f"mean_legal_actions={stats.mean_legal_actions:.2f}")
        for percentile in (50, 90, 95, 99, 100):
            print(f"p{percentile}_legal_actions={stats.percentile(percentile)}")


if __name__ == "__main__":
    main()
