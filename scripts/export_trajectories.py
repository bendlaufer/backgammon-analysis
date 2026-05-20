import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.mat import parse_match_text
from bg_rl.trajectory import match_to_compact_records, match_to_records, record_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validated checker decisions to JSONL.")
    parser.add_argument("mat_zip", help="Path to a zip containing .mat logs.")
    parser.add_argument("--out", default="artifacts/trajectories/checker_decisions.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="Maximum logs to export; 0 means all.")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--format", choices=("full", "compact"), default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.mat_zip).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logs = 0
    decisions = 0
    with zipfile.ZipFile(zip_path, "r") as zf, out_path.open("w", encoding="utf-8") as out:
        for name in zf.namelist():
            if not name.lower().endswith(".mat"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            try:
                parsed = parse_match_text(text, validate=True)
            except Exception as exc:
                raise RuntimeError(f"failed while exporting {name}") from exc
            records = (
                match_to_records(parsed, source_file=name)
                if args.format == "full"
                else match_to_compact_records(parsed, source_file=name)
            )
            for record in records:
                out.write(record_to_json(record) + "\n")
                decisions += 1
            logs += 1
            if args.progress_every > 0 and logs % args.progress_every == 0:
                print(f"Exported {decisions} checker decisions from {logs} logs...")
            if args.limit > 0 and logs >= args.limit:
                break

    print(f"Exported {decisions} checker decisions from {logs} logs to {out_path}.")


if __name__ == "__main__":
    main()
