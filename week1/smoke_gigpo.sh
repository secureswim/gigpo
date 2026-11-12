#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$PROJECT_ROOT/upstream-verl-agent"
RUN_DIR="$PROJECT_ROOT/runs/week1_smoke"
EXPECTED_SHA=f974dc5977908d6007ad178c9f3c48f0c7b39331

python3 "$PROJECT_ROOT/week1/check_baseline_scripts.py"
if ! command -v gpurun >/dev/null 2>&1; then
  printf 'ERROR: gpurun is required by this cluster. Run this script on the GPU server.\n' >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

{
  printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'upstream_commit=%s\n' "$(cat "$REPO/UPSTREAM_COMMIT")"
  printf 'expected_commit=%s\n' "$EXPECTED_SHA"
  printf 'python=%s\n' "$(python3 --version 2>&1)"
  printf 'git=%s\n' "$(git --version)"
  printf 'command=%s\n' "gpurun -g 2 bash examples/gigpo_trainer/run_alfworld.sh vllm trainer.total_epochs=1 trainer.test_freq=1 trainer.logger=['console'] trainer.experiment_name=week1_smoke"
  python3 -m pip freeze
} > "$RUN_DIR/metadata.txt"

cd "$REPO"
set -o pipefail
gpurun -g 2 bash examples/gigpo_trainer/run_alfworld.sh vllm \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  "trainer.logger=['console']" \
  trainer.experiment_name=week1_smoke 2>&1 | tee "$RUN_DIR/train.log"
