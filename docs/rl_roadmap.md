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
