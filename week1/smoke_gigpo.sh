#!/usr/bin/env bash
# Pipeline smoke test with the PAPER settings and the patched tree: one GiGPO
# training step with validation and a checkpoint, then a relaunch with a 2-step
# budget that must resume from the step-1 checkpoint.
#
# On Omni it runs inside a GPU job (never on the login node):
#   sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch smoke
#
# week1/check_smoke.py then verifies updates, validation, diagnostics, the
# environment-step counter, the checkpoint and the resume, and prints the
# seconds per training step used for the budget (RUNBOOK_OMNI.md, step 5).
# The success rate of this run means nothing.
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
