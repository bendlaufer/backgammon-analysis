# Browser Player

Run a local browser player against the current basic engine:

```bash
make serve-player
```

Open:

```text
http://127.0.0.1:8000
```

The server uses the behavior-cloning checkpoint at
`artifacts/bc-policy-full-local/model.pt` when present. If no checkpoint is
available, it falls back to a deterministic heuristic over legal moves.

This is a development UI for exercising the engine pipeline. It currently
supports checker play only; cube decisions and match play UI are future work.
