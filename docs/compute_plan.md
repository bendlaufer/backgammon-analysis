# Compute Plan

## Local Machine

Use local CPU for:

- unit tests
- parser changes
- 100-log sample exports
- small BC/value smoke runs
- quick full compact export when data is already local

Observed local results:

- full compact export: about 45 seconds for 987,811 decisions
- full legal-action stats with 4 workers: about 9 minutes
- BC training is CPU-bound by legal-action regeneration

## G2 CPU

Use G2 CPU resources for:

- repeated full-corpus legal-action stats
- full-corpus BC training
- afterstate feature cache generation
- hyperparameter sweeps for small CPU models

Lisbeth-style CPU resources are a better fit than GPUs for the current stage.
The workload is dominated by Python legal-action generation and feature
construction.

Use `cluster/g2/*.sub` as starting Slurm scripts.

## GPU

Use GPUs later for:

- larger state/action encoders
- batched afterstate value scoring
- self-play training with neural inference
- policy/value networks with enough tensor work to saturate a GPU

Do not request GPUs for the current MLP baselines unless profiling shows the
model forward/backward pass is the bottleneck.

## Totient

Totient is secondary for this project. It uses TORQUE/PBS and older system
software. It can run CPU diagnostics, but G2 is more appropriate for modern
Python/PyTorch workflows.
