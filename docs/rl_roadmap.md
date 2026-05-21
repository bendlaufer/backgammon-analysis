# Backgammon RL Roadmap

This project should treat the game implementation as an RL environment, not as
an eXtreme Gammon clone. The goal is to learn policy, value, and cube behavior
through self-play and targeted evaluation while keeping the game rules exact.

## Near-Term Milestones

1. Exact checker-play environment with immutable states, legal actions, and
   deterministic transitions.
2. Parser that converts `.mat` game logs into state/action trajectories for
   validation and optional behavior-cloning warm starts.
3. Neural policy/value model over legal actions and scalar/vector equity
   targets.
4. Self-play loop with chance-node dice sampling/enumeration and replay buffer.
5. Evaluation harness against fixed tactical, backgame, bearoff, opening, and
   cube-position suites.

## Self-Improvement Skeleton

The repository now has the first end-to-end RL loop:

- `bg_rl.rl.PolicyValueNet`: dual-head policy/value model.
- `bg_rl.rl.SearchMovePolicy`: search-compatible move policy that emits replay
  targets.
- `scripts/generate_rl_self_play.py`: writes self-play replay shards.
- `scripts/train_rl_policy_value.py`: trains a policy/value model from replay.
- `cluster/unicorn/rl_self_play_array_cpu.sub`: Slurm job array for many shards.
- `cluster/unicorn/rl_train_cpu.sub`: Slurm training job for one iteration.

The current search is intentionally shallow but replay-compatible with deeper
PUCT/MCTS. The next algorithmic upgrade is replacing the one-ply policy target
with chance-node-aware tree search over dice rolls and learned afterstate
values.

## Scale-Up Plan

1. Bootstrap: train BC/value models from human logs to avoid random early play.
2. Iteration 0: generate many CPU self-play shards with heuristic cube policy.
3. Train: fit the policy/value network on replay policy targets and game
   outcomes.
4. Promote: only advance a model when it beats the previous promoted model on a
   fixed seeded evaluation suite.
5. Upgrade search: add dice-chance expansion, cube decisions, and batched GPU
   inference once model evaluation dominates simulator cost.
6. Target XG weaknesses: maintain curated position suites from expert reports
   and compare rollout/value decisions across promoted models.

## Design Principles

- Keep rules exact and learning approximate.
- Use human games as bootstrap data, not as the final target.
- Prefer policy-guided search and uncertainty-aware rollouts over fixed
  hand-authored pruning intervals.
- Include cube and match context in state from the beginning.
- Make every training position reproducible from a canonical state/action log.

## Research Anchors

- Tesauro TD-Gammon: self-play temporal-difference learning.
- GNU Backgammon: neural evaluator, bearoff databases, rollout/evaluation
  conventions.
- XG public notes: search interval weaknesses, opening book and benchmark
  methodology.
