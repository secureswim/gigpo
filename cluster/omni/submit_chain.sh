#!/usr/bin/env bash
# Submit a training run as a chain of Slurm jobs, for runs longer than the
# partition's time limit. Each job resumes from the latest checkpoint; a job that
# finds the run already finished exits at once.
#
#   SEED=0 bash cluster/omni/submit_chain.sh <condition> <number_of_jobs> <RESOURCE FLAGS...>
#
# Example (the resource flags are placeholders; use the ones fixed in RUNBOOK step 3):
#   SEED=0 bash cluster/omni/submit_chain.sh gigpo 4 -p <partition> --gres=<gpu spec> --cpus-per-task=<n> --mem=<mem> --time=<limit>
#
# Prints the job IDs. Cancel the rest of a chain with: scancel <id> <id> ...
set -euo pipefail

CONDITION="${1:?condition}"
N="${2:?number of jobs}"
shift 2

case "$CONDITION" in
  gigpo|gigpo_threshold|gigpo_weighted) ;;
  *) echo "Only gigpo, gigpo_threshold and gigpo_weighted are allowed." >&2; exit 2 ;;
esac
if [[ "${SEED:-0}" != "0" ]]; then
  echo "Only SEED=0 is allowed by the three-experiment handoff." >&2; exit 2
fi
if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "number_of_jobs must be a positive integer." >&2; exit 2
fi
if [[ "$CONDITION" != gigpo ]]; then
  [[ -f results/diagnostic/thresholds.env ]] || { echo "Run the GiGPO diagnostic first: thresholds.env is missing." >&2; exit 2; }
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p slurm-logs

prev=""
ids=()
for ((i = 1; i <= N; i++)); do
  dep=()
  [[ -n "$prev" ]] && dep=(--dependency="afterany:$prev")
  prev=$(sbatch --parsable "${dep[@]}" --job-name="gigpo-$CONDITION-s${SEED:-0}" "$@" cluster/omni/job.sbatch "$CONDITION")
  prev="${prev%%;*}"
  ids+=("$prev")
done
echo "Submitted chain for $CONDITION seed ${SEED:-0}: ${ids[*]}"
echo "${ids[*]}" >> "slurm-logs/chains.txt"
