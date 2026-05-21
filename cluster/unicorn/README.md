# Unicorn Cluster Runs

Unicorn is the current Cornell Slurm cluster. Use these templates instead of
the older G2 templates when logging into `unicorn-login-01`.

The important differences from the old G2 notes:

- Anaconda path: `/share/apps/software/anaconda3`
- Apptainer module: available as `apptainer-1.4.0` / `apptainer-1.4.5`
- Slurm defaults are small: 1 CPU, 1 GB RAM, 4 hours, no GPU
- Always request CPU, memory, partition, and time explicitly

## One-Time Environment Setup

From the repo root on Unicorn:

```bash
cp cluster/unicorn/unicorn.env.example cluster/unicorn/unicorn.env
chmod 600 cluster/unicorn/unicorn.env
```

Edit `cluster/unicorn/unicorn.env` with real paths. This file is ignored by git
because `cluster/**/*.env` is ignored.

Create a conda environment:

```bash
source cluster/unicorn/unicorn.env
"$CONDA_BASE/bin/conda" create -p "$CONDA_ENV" python=3.12 -y
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `conda activate` complains that shell initialization is missing, run:

```bash
/share/apps/software/anaconda3/bin/conda init
```

Then start a new shell and retry.

## Smoke Test

```bash
source cluster/unicorn/unicorn.env
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python -m unittest discover -s tests
```

## Submit Jobs

Use the lowest-priority partition that meets the job. Override the partition on
submission if needed:

```bash
source cluster/unicorn/unicorn.env
sbatch --requeue --partition="${UNICORN_PARTITION:-default_partition}" cluster/unicorn/export_compact_cpu.sub
sbatch --requeue --partition="${UNICORN_PARTITION:-default_partition}" cluster/unicorn/stats_cpu.sub
sbatch --requeue --partition="${UNICORN_PARTITION:-default_partition}" cluster/unicorn/train_value_cpu.sub
sbatch --requeue --partition="${UNICORN_PARTITION:-default_partition}" cluster/unicorn/train_bc_cpu.sub
sbatch --requeue --partition="${UNICORN_PARTITION:-default_partition}" cluster/unicorn/generate_self_play_cpu.sub
```

Monitor:

```bash
squeue --me
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList
```

## GPU

Do not request GPUs for the current MLP baselines. Use GPU later for larger
neural encoders or batched self-play inference.
