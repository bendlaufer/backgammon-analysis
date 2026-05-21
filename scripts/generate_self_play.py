import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.engine import BasicEngine
from bg_rl.self_play import RandomPolicy, build_cube_policy, play_game, self_play_record_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cube-aware checker self-play trajectories.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--out", default="artifacts/self-play/cube_self_play.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plies", type=int, default=2000)
    parser.add_argument("--policy0", choices=("engine", "random"), default="engine")
    parser.add_argument("--policy1", choices=("engine", "random"), default="engine")
    parser.add_argument("--cube-policy0", choices=("none", "heuristic"), default="none")
    parser.add_argument("--cube-policy1", choices=("none", "heuristic"), default="none")
    parser.add_argument("--model-path", default="artifacts/bc-policy-full-local/model.pt")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    policy0 = build_policy(args.policy0, args.model_path, rng)
    policy1 = build_policy(args.policy1, args.model_path, rng)
    cube_policy0 = build_cube_policy(args.cube_policy0)
    cube_policy1 = build_cube_policy(args.cube_policy1)

    total_decisions = 0
    wins = {0: 0, 1: 0, None: 0}
    with out_path.open("w", encoding="utf-8") as fp:
        for game_index in range(args.games):
            game = play_game(
                game_index=game_index,
                policy0=policy0,
                policy1=policy1,
                cube_policy0=cube_policy0,
                cube_policy1=cube_policy1,
                rng=rng,
                max_plies=args.max_plies,
            )
            wins[game.winner] += 1
            total_decisions += len(game.decisions)
            for decision in game.decisions:
                fp.write(self_play_record_to_json(decision) + "\n")
            if args.progress_every > 0 and (game_index + 1) % args.progress_every == 0:
                print(f"games={game_index + 1} decisions={total_decisions} wins={wins}")

    print(f"Wrote {total_decisions} decisions from {args.games} games to {out_path}.")
    print(f"Wins: {wins}")


def build_policy(name: str, model_path: str, rng: random.Random):
    if name == "random":
        return RandomPolicy(rng)
    return BasicEngine(model_path)


if __name__ == "__main__":
    main()
