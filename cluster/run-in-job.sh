#!/usr/bin/env bash
set -euo pipefail
JOB=${1:?job id required}
[[ $JOB =~ ^[1-9][0-9]*$ ]] || exit 2
ROOT=/home/vansh23580/rl-project
status="$ROOT/logs/gigpo-$JOB.status"
trap 'result=$?; printf "exit=%s date=%s\n" "$result" "$(date -Is)" >> "$status"' EXIT
printf 'started=%s\n' "$(date -Is)" > "$status"
OMNI_STEP_CPUS=1 bash "$ROOT/cluster/job-step.sh" "$JOB" bash \
  "$ROOT/cluster/inspect-gpu.sh" "$ROOT/envs/gigpo" --smoke \
  > "$ROOT/logs/gigpo-$JOB-cuda.log" 2>&1
bash "$ROOT/cluster/job-step.sh" "$JOB" bash \
  "$ROOT/gigpo/cluster/smoke-omni.sh" \
  > "$ROOT/logs/gigpo-$JOB.log" 2>&1
