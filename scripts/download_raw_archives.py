import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download full raw Arkadium Backgammon ZIP archives.")
    parser.add_argument("--repo-id", default="ArkadiumInc/ArkadiumBackgammon")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--which",
        choices=["gamelogs", "gamelogs_with_images", "both"],
        default="gamelogs",
        help="Which raw archive(s) to download.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download if already present.")
    return parser.parse_args()


def pick_file(repo_files: list[str], contains: str) -> str:
    matches = [f for f in repo_files if f.lower().endswith(".zip") and contains in f.lower()]
    if not matches:
        raise FileNotFoundError(f"No ZIP file found containing '{contains}' in dataset repo files.")
    matches.sort()
    return matches[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")

    targets: list[str] = []
    if args.which in {"gamelogs", "both"}:
        targets.append(pick_file(repo_files, "gamelogs_001"))
    if args.which in {"gamelogs_with_images", "both"}:
        targets.append(pick_file(repo_files, "gamelogs_and_images_002"))

    for remote_file in targets:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=remote_file,
            repo_type="dataset",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            force_download=args.force,
        )
        print(f"Downloaded: {remote_file}")
        print(f"Local path: {local_path}")


if __name__ == "__main__":
    main()
