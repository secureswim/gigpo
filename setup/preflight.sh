#!/usr/bin/env bash
# Checks to run on the GPU machine before any long run (project root):
#
#   bash setup/preflight.sh
#
# Fails fast with a clear message. Writes runs/preflight.txt.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p runs
REPORT=runs/preflight.txt
: > "$REPORT"
fail=0

check() {  # check "<name>" <command...>
  local name="$1"; shift
  if out="$("$@" 2>&1)"; then
    echo "PASS  $name" | tee -a "$REPORT"
  else
    echo "FAIL  $name" | tee -a "$REPORT"
    echo "$out" | tail -20 | sed 's/^/      /' | tee -a "$REPORT"
    fail=1
  fi
}

check "snapshot checksums and matched baseline settings" python3 week1/check_baseline_scripts.py
check "patched tree present" test -f work/verl-agent-ext/gigpo/gigpo_ext.py
check "CPU unit tests" python3 -m pytest -q tests
check "nvidia-smi" nvidia-smi
check "torch sees CUDA" python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
check "vllm 0.8.2 import" python3 -c "import vllm; assert vllm.__version__.startswith('0.8.2'), vllm.__version__"
check "flash-attn import" python3 -c "import flash_attn"
check "verl resolves to patched tree" python3 -c "import verl, os; p=os.path.dirname(verl.__file__); assert 'verl-agent-ext' in p, p"
check "ALFWORLD_DATA set and downloaded" bash -c 'test -n "${ALFWORLD_DATA:-}" && test -d "$ALFWORLD_DATA/json_2.1.1"'
check "ALFWorld reset/step" python3 week1/check_alfworld.py
check "model available" python3 -c "
import os
from transformers import AutoTokenizer
p = os.environ.get('MODEL_PATH', 'Qwen/Qwen2.5-1.5B-Instruct')
AutoTokenizer.from_pretrained(p); print(p)"
check "disk space (>= 100 GB free in runs/)" bash -c '[ "$(df -Pk runs | awk "NR==2{print \$4}")" -ge $((100*1024*1024)) ]'

{
  echo "--- versions"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null
  python3 -c "import torch,vllm,transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__,'transformers',transformers.__version__)" 2>/dev/null
} >> "$REPORT"

if [[ $fail -ne 0 ]]; then
  echo "Preflight FAILED. See $REPORT" >&2
  exit 1
fi
echo "Preflight passed. Next: bash week1/smoke_gigpo.sh"
