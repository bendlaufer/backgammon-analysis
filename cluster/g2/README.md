# G2 Cluster Runs

Use G2 for repeated full-corpus CPU jobs and larger PyTorch training. For this
project, CPU resources are the first useful cluster target because legal-action
generation and afterstate featurization are Python/CPU-heavy. GPUs become useful
after we move to larger neural encoders or self-play training.

## Recommended Storage

- Keep code in home or clone it into a project folder.
- Keep data/artifacts outside home because home has a small quota.
- For CPU jobs, use a shared project path such as `/share/pierson/...` if you
  have permission.
- For GPU jobs later, copy active datasets to node-local `/scratch/...` before
  training.

Set these paths in submitted jobs:

```bash
export PROJECT_DIR=/path/to/backgammon-analysis
export BG_DATA_DIR=/share/pierson/YOUR_PROJECT/backgammon
export CONDA_ENV=/share/pierson/conda_virtualenvs/YOUR_NETID_backgammon
```

Use `cluster/g2/g2.env.example` as a template. Copy it to
`cluster/g2/g2.env` and fill in real values there. The filled `.env` file is
ignored by git and should not be committed.

Expected files under `BG_DATA_DIR`:

```text
data/Arkadium_Backgammon_full_data_gamelogs_001.zip
artifacts/trajectories/checker_decisions_compact_full.jsonl
```

## Environment

Create a conda environment on the cluster, preferably outside home if packages
will be large:

```bash
conda create --prefix /share/pierson/conda_virtualenvs/YOUR_NETID_backgammon python=3.10
conda activate /share/pierson/conda_virtualenvs/YOUR_NETID_backgammon
pip install -r requirements.txt
```

Then submit jobs from the repo root:

```bash
sbatch --requeue cluster/g2/export_compact_cpu.sub
sbatch --requeue cluster/g2/stats_cpu.sub
sbatch --requeue cluster/g2/train_value_cpu.sub
sbatch --requeue cluster/g2/train_bc_cpu.sub
```

Use `squeue -u YOUR_NETID` to monitor.

See `cluster/g2/setup_checklist.md` for a full setup checklist.

## Current Recommendation

Start with CPU jobs on `pierson` or a CPU partition. Do not request GPUs for the
current MLP baselines; they are bottlenecked by legal-action generation and
feature construction, not matrix multiplication.
