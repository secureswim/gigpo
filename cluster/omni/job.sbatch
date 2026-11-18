#!/usr/bin/env bash
#SBATCH --job-name=gigpo
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --open-mode=append
# One Slurm batch job. Submit from the repository root; partition, GPU, CPUs,
# memory and time limit go on the sbatch command line (RUNBOOK_OMNI.md, step 3):
#
#   sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch preflight
#   sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch smoke
#   SEED=0 sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch gigpo
#   SEED=0 sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch gigpo_threshold   # needs results/diagnostic/thresholds.env
#   SEED=0 sbatch <RESOURCE FLAGS> cluster/omni/job.sbatch gigpo_weighted    # needs results/diagnostic/thresholds.env
#
# Environment variables on the sbatch command line are passed into the job
# (Slurm's default --export=ALL). The job exits when its work ends, so the
# allocation is released immediately. A training run that hits the time limit
# is resumed by submitting the same command again (or use submit_chain.sh).
set -euo pipefail

MODE="${1:?usage: job.sbatch <preflight|smoke|gigpo|gigpo_threshold|gigpo_weighted>}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root with sbatch}"
[[ -f cluster/omni/env.sh ]] || { echo "Submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source cluster/omni/env.sh
echo "job=$SLURM_JOB_ID mode=$MODE node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Slurm ends all of this job's processes on exit. Do not run `ray stop`: it would
# also kill Ray in any other job of yours on the same node.
cleanup() { rm -rf "${RAY_TMPDIR:-/nonexistent}"; echo "end=$(date -Is)"; }
trap cleanup EXIT

case "$MODE" in
  preflight)
    bash setup/preflight.sh --gpu ;;
  smoke)
    bash week1/smoke_gigpo.sh ;;
  gigpo|gigpo_threshold|gigpo_weighted)
    if [[ "$MODE" == gigpo_threshold || "$MODE" == gigpo_weighted ]]; then
      [[ -f results/diagnostic/thresholds.env ]] || { echo "results/diagnostic/thresholds.env missing (RUNBOOK step 8)" >&2; exit 1; }
      # shellcheck disable=SC1091
      source results/diagnostic/thresholds.env
    fi
    RUN_DIR="$RUN_ROOT/${MODE}_seed${SEED:-0}${RUN_TAG:+_$RUN_TAG}"
    if python3 analysis/progress.py --done "$RUN_DIR" "${TOTAL_EPOCHS:-150}"; then
      echo "Run $RUN_DIR already reached step ${TOTAL_EPOCHS:-150}; nothing to do."
      exit 0
    fi
    bash scripts/run_condition.sh "$MODE" ;;
  *)
    echo "Unknown mode $MODE" >&2; exit 2 ;;
esac
