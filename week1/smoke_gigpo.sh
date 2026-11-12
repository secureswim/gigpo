#!/usr/bin/env bash
# Pipeline smoke test on the GPU machine: 2 training steps of original GiGPO,
# validation and a checkpoint after each step, then a resume check.
#
#   bash week1/smoke_gigpo.sh
#
# Part 1 trains step 1 and saves a checkpoint. Part 2 relaunches the same run
# with TOTAL_EPOCHS=2, which must resume from step 1 rather than start again.
# week1/check_smoke.py then verifies the log, the diagnostics and the resume.
# The score of this run means nothing; it only verifies the pipeline.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export RUN_TAG=smoke WANDB=0 TEST_FREQ=1 SAVE_FREQ=1 CKPT_KEEP=1 SEED="${SEED:-0}"
RUN_DIR="$RUN_ROOT/gigpo_seed${SEED}_smoke"

if [[ -d "$RUN_DIR" ]]; then
  echo "$RUN_DIR exists from an earlier smoke run; move or delete it first." >&2
  exit 1
fi

python3 "$PROJECT_ROOT/week1/check_baseline_scripts.py"

start=$(date +%s)
TOTAL_EPOCHS=1 bash "$PROJECT_ROOT/scripts/run_condition.sh" gigpo
mid=$(date +%s)
TOTAL_EPOCHS=2 bash "$PROJECT_ROOT/scripts/run_condition.sh" gigpo
end=$(date +%s)
printf 'step1_seconds=%s\nresume_step2_seconds=%s\n' $((mid - start)) $((end - mid)) >> "$RUN_DIR/launches.txt"

python3 "$PROJECT_ROOT/week1/check_smoke.py" "$RUN_DIR"
