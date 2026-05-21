# Benchmark Suite

The benchmark suite is the repository's curated set of difficult positions for
measuring whether a trained engine is learning more than average human-game
patterns.

The suite intentionally separates three stages:

- `active`: a parsed `BackgammonState` is present and the position can be scored
  by our engine today.
- `opaque`: the source gives an XGID or diagram, but we have not decoded it into
  our internal state yet.
- `candidate`: the source identifies useful positions or classes, but we still
  need exact board states.

Run validation:

```bash
python scripts/validate_benchmarks.py
```

Score currently active positions:

```bash
python scripts/score_benchmarks.py
```

## Source Policy

Every benchmark that claims XG disagreement must include:

- source title and URL
- line references or enough context to recover the claim
- exact position representation, preferably XGID
- decision type: checker, cube, equity, or mixed
- the reported XG choice and expert/rollout alternative when available
- caveat notes when the source itself is uncertain

The first source-backed seeds are:

- eXtreme Gammon's own search-interval article, including a concrete 62
  containment example where a rollout shows a missed play.
- Backgammon Forums discussion of XG weaknesses in snake/deep-containment and
  backgame positions, including XGIDs supplied by Frank Berger/BGBlitz.
- Fortuitous Press's Backgammon Boot Camp errata list, which identifies many
  backgame/cube revisions where XG2 contradicts older Snowie-era conclusions.

## Near-Term Work

1. Implement XGID decoding into `BackgammonState`.
2. Convert opaque XGID benchmarks into active benchmark rows.
3. Add cube/equity scoring once match-equity and cube models mature.
4. Add an expert-reviewed `expected_action` field only when the claim is strong
   enough to treat as a target, not merely a curiosity.
5. Keep benchmark reports under `artifacts/benchmark_reports/` so model
   generations can be compared over time.
