PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup install register-kernel freeze preview download-raw train-lm generate-lm test validate-mat export-trajectories export-compact-trajectories trajectory-stats train-bc train-value generate-self-play generate-rl-self-play train-rl serve-player clean-venv

help:
	@echo "Available targets:"
	@echo "  make setup      Create venv and install dependencies"
	@echo "  make install    Install dependencies into existing venv"
	@echo "  make register-kernel Register venv as Jupyter kernel"
	@echo "  make freeze     Write exact installed versions to requirements.txt"
	@echo "  make preview    Run dataset preview script"
	@echo "  make download-raw Download full raw non-image zip"
	@echo "  make train-lm   Train small next-token transformer on moves"
	@echo "  make generate-lm Generate continuation from trained model"
	@echo "  make test       Run unit tests"
	@echo "  make validate-mat Validate sample raw .mat logs against bg_rl rules"
	@echo "  make export-trajectories Export sample validated checker decisions to JSONL"
	@echo "  make export-compact-trajectories Export compact checker decisions to JSONL"
	@echo "  make trajectory-stats Summarize exported trajectory JSONL rows"
	@echo "  make train-bc   Train a small behavior-cloning policy baseline"
	@echo "  make train-value Train a small supervised value baseline"
	@echo "  make generate-self-play Generate cube-aware self-play trajectories"
	@echo "  make generate-rl-self-play Generate policy/value RL replay trajectories"
	@echo "  make train-rl    Train policy/value network from RL replay"
	@echo "  make serve-player Run browser player at http://127.0.0.1:8000"
	@echo "  make clean-venv Remove local virtual environment"

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install:
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

register-kernel:
	$(VENV_PYTHON) -m ipykernel install --user --name backgammon-analysis --display-name "Python (backgammon-analysis)"

freeze:
	$(VENV_PIP) freeze > requirements.txt

preview:
	$(VENV_PYTHON) scripts/load_dataset_preview.py

download-raw:
	$(VENV_PYTHON) scripts/download_raw_archives.py --which gamelogs --out-dir data

train-lm:
	$(VENV_PYTHON) scripts/train_moves_lm.py

generate-lm:
	$(VENV_PYTHON) scripts/generate_moves.py

test:
	$(VENV_PYTHON) -m unittest discover -s tests

validate-mat:
	$(VENV_PYTHON) scripts/validate_mat_logs.py data/Arkadium_Backgammon_full_data_gamelogs_001.zip --limit 100

export-trajectories:
	$(VENV_PYTHON) scripts/export_trajectories.py data/Arkadium_Backgammon_full_data_gamelogs_001.zip --limit 100

export-compact-trajectories:
	$(VENV_PYTHON) scripts/export_trajectories.py data/Arkadium_Backgammon_full_data_gamelogs_001.zip --format compact --trust-parser-validation --limit 100 --out artifacts/trajectories/checker_decisions_compact.jsonl

trajectory-stats:
	$(VENV_PYTHON) scripts/trajectory_stats.py

train-bc:
	$(VENV_PYTHON) scripts/train_bc_policy.py --max-samples 5000 --epochs 5

train-value:
	$(VENV_PYTHON) scripts/train_value_model.py --max-samples 5000 --epochs 5

generate-self-play:
	$(VENV_PYTHON) scripts/generate_self_play.py --games 10 --cube-policy0 heuristic --cube-policy1 heuristic

generate-rl-self-play:
	$(VENV_PYTHON) scripts/generate_rl_self_play.py --games 10 --out artifacts/rl-self-play/shard_000.jsonl

train-rl:
	$(VENV_PYTHON) scripts/train_rl_policy_value.py --data artifacts/rl-self-play/*.jsonl --epochs 2

serve-player:
	$(VENV_PYTHON) web/server.py --host 127.0.0.1 --port 8000

clean-venv:
	rm -rf $(VENV_DIR)
