# G2 Setup Checklist

This checklist intentionally uses placeholders only. Do not commit real NetIDs,
private storage paths, access tokens, SSH keys, or filled environment files.

## Local Machine

1. Commit or stash code changes locally.
2. Make sure large artifacts remain ignored:
   - `data/`
   - `artifacts/`
   - `*.jsonl`
   - `*.zip`
3. Connect to the Cornell VPN.
4. Log in with your own NetID:

```bash
ssh <YOUR_NETID>@g2-login.coecis.cornell.edu
```

## Cluster Environment

1. Create a private tmp directory:

```bash
mkdir -p /share/pierson/tmp_directory_location_please_read_readme/<YOUR_NETID>_tmp
chmod 700 /share/pierson/tmp_directory_location_please_read_readme/<YOUR_NETID>_tmp
```

2. Clone or transfer the repo. Prefer Git for code and `scp`/`rsync` for large
   local artifacts:

```bash
git clone <REPO_URL> ~/backgammon-analysis
```

3. Create or activate a conda environment outside home if possible:

```bash
conda create --prefix /share/pierson/conda_virtualenvs/<YOUR_NETID>_backgammon python=3.10
conda activate /share/pierson/conda_virtualenvs/<YOUR_NETID>_backgammon
cd ~/backgammon-analysis
pip install -r requirements.txt
```

4. Copy the env template and fill it locally:

```bash
cp cluster/g2/g2.env.example cluster/g2/g2.env
chmod 600 cluster/g2/g2.env
```

Edit `cluster/g2/g2.env` with real paths. This file is ignored by git.

## Data Placement

Create the data/artifact directories in shared storage:

```bash
mkdir -p "$BG_DATA_DIR/data"
mkdir -p "$BG_DATA_DIR/artifacts/trajectories"
mkdir -p "$BG_DATA_DIR/artifacts/models"
```

Transfer the raw archive to:

```text
$BG_DATA_DIR/data/Arkadium_Backgammon_full_data_gamelogs_001.zip
```

Do not put large data in `/home/<YOUR_NETID>`.

## Smoke Test

From the repo root on G2:

```bash
source cluster/g2/g2.env
python -m unittest discover -s tests
python scripts/export_trajectories.py \
  "$BG_DATA_DIR/data/Arkadium_Backgammon_full_data_gamelogs_001.zip" \
  --format compact \
  --trust-parser-validation \
  --limit 10 \
  --out "$BG_DATA_DIR/artifacts/trajectories/checker_decisions_compact_smoke.jsonl"
```

## Submit Jobs

Submit CPU jobs from the repo root:

```bash
source cluster/g2/g2.env
sbatch --requeue cluster/g2/export_compact_cpu.sub
sbatch --requeue cluster/g2/stats_cpu.sub
sbatch --requeue cluster/g2/train_value_cpu.sub
sbatch --requeue cluster/g2/train_bc_cpu.sub
```

Monitor jobs:

```bash
squeue -u <YOUR_NETID>
```

## Before Committing

Run these checks locally or on the cluster:

```bash
git status --short
git diff --cached
```

Confirm these are not staged:

- `cluster/g2/g2.env`
- files containing real NetIDs
- SSH keys
- access tokens
- large data or model artifacts
