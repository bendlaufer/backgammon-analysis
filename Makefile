PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup install register-kernel freeze preview download-raw train-lm generate-lm clean-venv

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

clean-venv:
	rm -rf $(VENV_DIR)
