import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.rl import SearchConfig, SearchMovePolicy, load_policy_value_model, rl_record_to_json
from bg_rl.self_play import build_cube_policy, play_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate policy/value RL self-play replay shards.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out", default="artifacts/rl-self-play/shard_000.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--cube-policy", choices=("none", "heuristic"), default="heuristic")
    parser.add_argument("--max-plies", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    model = load_policy_value_model(args.model_path or None, hidden_dim=args.hidden_dim)
    policy = SearchMovePolicy(
        model,
        rng=rng,
        config=SearchConfig(simulations=args.simulations, temperature=args.temperature),
    )
    cube_policy = build_cube_policy(args.cube_policy)

    total_records = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for game_index in range(args.games):
            start_history = len(policy.history)
            game = play_game(
                game_index=game_index,
                policy0=policy,
                policy1=policy,
                cube_policy0=cube_policy,
                cube_policy1=cube_policy,
                rng=rng,
                max_plies=args.max_plies,
            )
            game_searches = policy.history[start_history:]
            checker_decisions = [decision for decision in game.decisions if decision.phase == "checker"]
            if len(checker_decisions) != len(game_searches):
                raise RuntimeError("search history and checker decisions diverged")
            for decision, result in zip(checker_decisions, game_searches):
                assert decision.dice is not None
                fp.write(
                    rl_record_to_json(
                        game_index=game_index,
                        ply=decision.ply,
                        player=decision.player,
                        state=decision.state,
                        dice=decision.dice,
                        result=result,
                        reward=float(decision.reward or 0.0),
                        points_reward=int(decision.points_reward or 0),
                    )
                    + "\n"
                )
                total_records += 1
            if args.progress_every > 0 and (game_index + 1) % args.progress_every == 0:
                print(f"games={game_index + 1} records={total_records}")

    print(f"Wrote {total_records} replay records from {args.games} games to {out_path}.")


if __name__ == "__main__":
    main()
