from datasets import load_dataset


def main() -> None:
    # Load game logs preview split.
    ds = load_dataset("ArkadiumInc/ArkadiumBackgammon", "gamelogs")

    # Load game logs with board screenshots preview split.
    ds_img = load_dataset("ArkadiumInc/ArkadiumBackgammon", "gamelogs_with_images")

    match = ds["train"][0]

    print(f"Players: {match['player1']} vs {match['player2']}")
    print(f"Date: {match['event_date']}")
    print(f"Result: {match['result']}")
    print(f"\nMoves:\n{match['moves']}")
    print(f"\nImage-enabled records in train split: {len(ds_img['train'])}")


if __name__ == "__main__":
    main()
