import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.mat import parse_match_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate .mat logs against bg_rl rules.")
    parser.add_argument("mat_zip", help="Path to a zip containing .mat logs.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum logs to validate.")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.mat_zip).expanduser()
    parsed_logs = 0
    decisions = 0

    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".mat"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            try:
                parsed = parse_match_text(text, validate=True)
            except Exception as exc:
                raise RuntimeError(f"failed while validating {name}") from exc
            parsed_logs += 1
            decisions += len(parsed.decisions)
            if args.progress_every > 0 and parsed_logs % args.progress_every == 0:
                print(f"Validated {parsed_logs} logs with {decisions} checker decisions...")
            if args.limit > 0 and parsed_logs >= args.limit:
                break

    print(f"Validated {parsed_logs} logs with {decisions} checker decisions.")


if __name__ == "__main__":
    main()
