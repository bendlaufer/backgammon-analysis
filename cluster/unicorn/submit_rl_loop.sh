#!/bin/bash
set -euo pipefail

ITERATIONS=1
PARTITION="${UNICORN_PARTITION:-garg}"
ARRAY="0-15"
GAMES_PER_SHARD=1000
EVAL_GAMES=100
TRAIN_EPOCHS=3
MAX_SAMPLES=0
MIN_WIN_RATE=0.50
MIN_POINT_MARGIN=-0.05
SIMULATIONS=1
TEMPERATURE=1.0

usage() {
  cat <<'EOF'
Usage: bash cluster/unicorn/submit_rl_loop.sh [options]

Options:
  --iterations N          Number of RL iterations to submit (default: 1)
  --partition NAME        Slurm partition (default: $UNICORN_PARTITION or garg)
  --array SPEC            Self-play array spec, e.g. 0-15 (default: 0-15)
  --games-per-shard N     Games per self-play array task (default: 1000)
  --eval-games N          Evaluation games per promotion gate (default: 100)
  --train-epochs N        Training epochs per iteration (default: 3)
  --max-samples N         Max replay samples for training, 0 means all (default: 0)
  --min-win-rate X        Promotion win-rate threshold (default: 0.50)
  --min-point-margin X    Promotion point margin threshold (default: -0.05)
  --simulations N         Search simulations placeholder (default: 1)
  --temperature X         Self-play action temperature (default: 1.0)
  -h, --help              Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --array) ARRAY="$2"; shift 2 ;;
    --games-per-shard) GAMES_PER_SHARD="$2"; shift 2 ;;
    --eval-games) EVAL_GAMES="$2"; shift 2 ;;
    --train-epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --min-win-rate) MIN_WIN_RATE="$2"; shift 2 ;;
    --min-point-margin) MIN_POINT_MARGIN="$2"; shift 2 ;;
    --simulations) SIMULATIONS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -f cluster/unicorn/unicorn.env ]; then
  source cluster/unicorn/unicorn.env
fi

: "${PROJECT_DIR:?Set PROJECT_DIR in cluster/unicorn/unicorn.env or the environment}"
: "${BG_DATA_DIR:?Set BG_DATA_DIR in cluster/unicorn/unicorn.env or the environment}"

PROMOTED_MODEL="$BG_DATA_DIR/artifacts/rl-policy-value/promoted/model.pt"
mkdir -p "$BG_DATA_DIR/artifacts/rl-submissions"
SUMMARY="$BG_DATA_DIR/artifacts/rl-submissions/loop_$(date +%Y%m%d_%H%M%S).tsv"
printf "iteration\tself_play_job\ttrain_job\teval_job\n" > "$SUMMARY"

dependency=()
for ((i = 0; i < ITERATIONS; i++)); do
  iter=$(printf "%03d" "$i")
  allow_no_baseline=0
  if [ "$i" -eq 0 ] && [ ! -s "$PROMOTED_MODEL" ]; then
    allow_no_baseline=1
  fi

  self_play_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --partition="$PARTITION" \
    --array="$ARRAY" \
    --export=ALL,RL_ITERATION="$iter",RL_GAMES_PER_SHARD="$GAMES_PER_SHARD",RL_MODEL_PATH="$PROMOTED_MODEL",RL_SIMULATIONS="$SIMULATIONS",RL_TEMPERATURE="$TEMPERATURE" \
    cluster/unicorn/rl_self_play_array_cpu.sub)

  train_job=$(sbatch --parsable \
    --dependency=afterok:"$self_play_job" \
    --partition="$PARTITION" \
    --export=ALL,RL_ITERATION="$iter",RL_INIT_MODEL="$PROMOTED_MODEL",RL_EPOCHS="$TRAIN_EPOCHS",RL_MAX_SAMPLES="$MAX_SAMPLES",RL_SIMULATIONS="$SIMULATIONS" \
    cluster/unicorn/rl_train_cpu.sub)

  eval_job=$(sbatch --parsable \
    --dependency=afterok:"$train_job" \
    --partition="$PARTITION" \
    --export=ALL,RL_ITERATION="$iter",RL_BASELINE_MODEL="$PROMOTED_MODEL",RL_PROMOTE_DIR="$BG_DATA_DIR/artifacts/rl-policy-value/promoted",RL_EVAL_GAMES="$EVAL_GAMES",RL_MIN_WIN_RATE="$MIN_WIN_RATE",RL_MIN_POINT_MARGIN="$MIN_POINT_MARGIN",RL_ALLOW_NO_BASELINE="$allow_no_baseline",RL_SIMULATIONS="$SIMULATIONS" \
    cluster/unicorn/rl_evaluate_cpu.sub)

  printf "%s\t%s\t%s\t%s\n" "$iter" "$self_play_job" "$train_job" "$eval_job" | tee -a "$SUMMARY"
  dependency=(--dependency=afterok:"$eval_job")
done

echo "Submission summary: $SUMMARY"
