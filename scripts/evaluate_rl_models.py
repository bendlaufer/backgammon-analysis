import argparse
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.rl import SearchConfig, SearchMovePolicy, load_policy_value_model
from bg_rl.self_play import build_cube_policy, play_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and optionally promote an RL policy/value model.")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--report", default="artifacts/rl-eval/report.json")
    parser.add_argument("--promote-dir", default="")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--min-win-rate", type=float, default=0.50)
    parser.add_argument("--min-point-margin", type=float, default=-0.05)
    parser.add_argument("--allow-no-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_model)
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate model does not exist: {candidate_path}")

    baseline_path = Path(args.baseline_model) if args.baseline_model else None
    has_baseline = baseline_path is not None and baseline_path.exists()
    if not has_baseline and not args.allow_no_baseline:
        raise FileNotFoundError(f"baseline model does not exist: {baseline_path}")

    metrics = evaluate(
        candidate_path=candidate_path,
        baseline_path=baseline_path if has_baseline else None,
        games=args.games,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        simulations=args.simulations,
    )
    passed = metrics["has_baseline"] is False or (
        metrics["candidate_win_rate"] >= args.min_win_rate
        and metrics["candidate_point_margin_per_game"] >= args.min_point_margin
    )
    metrics["promotion_passed"] = passed
    metrics["min_win_rate"] = args.min_win_rate
    metrics["min_point_margin"] = args.min_point_margin

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if passed and args.promote_dir:
        promote_dir = Path(args.promote_dir)
        promote_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, promote_dir / "model.pt")
        shutil.copy2(report_path, promote_dir / "last_promotion_report.json")
        print(f"Promoted candidate to {promote_dir / 'model.pt'}")
    elif not passed:
        raise SystemExit("candidate did not pass promotion gate")


def evaluate(
    *,
    candidate_path: Path,
    baseline_path: Path | None,
    games: int,
    seed: int,
    hidden_dim: int,
    temperature: float,
    simulations: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    candidate_model = load_policy_value_model(candidate_path, hidden_dim=hidden_dim)
    baseline_model = load_policy_value_model(baseline_path, hidden_dim=hidden_dim)
    config = SearchConfig(simulations=simulations, temperature=temperature)
    cube_policy = build_cube_policy("heuristic")

    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    candidate_points = 0
    baseline_points = 0
    plies = 0

    for game_index in range(games):
        candidate_player = game_index % 2
        candidate_policy = SearchMovePolicy(candidate_model, rng=rng, config=config)
        baseline_policy = SearchMovePolicy(baseline_model, rng=rng, config=config)
        if candidate_player == 0:
            policy0, policy1 = candidate_policy, baseline_policy
        else:
            policy0, policy1 = baseline_policy, candidate_policy

        game = play_game(
            game_index=game_index,
            policy0=policy0,
            policy1=policy1,
            cube_policy0=cube_policy,
            cube_policy1=cube_policy,
            rng=rng,
        )
        plies += game.plies
        if game.winner is None:
            draws += 1
        elif game.winner == candidate_player:
            candidate_wins += 1
            candidate_points += game.points_won
        else:
            baseline_wins += 1
            baseline_points += game.points_won

    decisive_games = max(1, candidate_wins + baseline_wins)
    return {
        "candidate_model": str(candidate_path),
        "baseline_model": str(baseline_path) if baseline_path else "",
        "has_baseline": baseline_path is not None and baseline_path.exists(),
        "games": games,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "candidate_win_rate": candidate_wins / decisive_games,
        "candidate_points": candidate_points,
        "baseline_points": baseline_points,
        "candidate_point_margin_per_game": (candidate_points - baseline_points) / max(1, games),
        "avg_plies": plies / max(1, games),
    }


if __name__ == "__main__":
    main()
