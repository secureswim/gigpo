#!/usr/bin/env bash
# Preflight checks. Run from the repository root.
#
#   bash setup/preflight.sh --login   # login node, no GPU (called by cluster/omni/prepare.sh)
#   bash setup/preflight.sh --gpu     # inside a GPU job (sbatch ... cluster/omni/job.sbatch preflight)
#
# Prints PASS/FAIL per check, writes runs/preflight_<mode>.txt, exits 1 on any FAIL.
set -uo pipefail

MODE="${1:---login}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source cluster/omni/env.sh >/dev/null 2>&1 || true
mkdir -p runs
REPORT="runs/preflight_${MODE#--}.txt"
{ echo "preflight $MODE $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) slurm_job=${SLURM_JOB_ID:-none}"; } > "$REPORT"
fail=0

check() {  # check "<name>" <command...>
  local name="$1"; shift
  local out
  if out="$("$@" 2>&1)"; then
    echo "PASS  $name" | tee -a "$REPORT"
  else
    echo "FAIL  $name" | tee -a "$REPORT"
    echo "$out" | tail -20 | sed 's/^/      /' | tee -a "$REPORT"
    fail=1
  fi
}

# --- checks valid on any node ---
check "snapshot checksums and matched baseline settings" python3 week1/check_baseline_scripts.py
check "patched tree built" test -f "$EXT_TREE/gigpo/gigpo_ext.py"
check "verl resolves to the patched tree" python3 -c "import verl, os; p=os.path.dirname(verl.__file__); assert 'verl-agent-ext' in p, p"
check "model snapshot present (MODEL_PATH)" test -f "$MODEL_PATH/config.json"
check "ALFWorld data downloaded (ALFWORLD_DATA)" test -d "$ALFWORLD_DATA/json_2.1.1"
check "placeholder parquet files" bash -c 'test -f "$HOME/data/verl-agent/text/train.parquet" && test -f "$HOME/data/verl-agent/text/test.parquet"'
check "disk: >= 100 GB free for runs/" bash -c '[ "$(df -Pk runs | awk "NR==2{print \$4}")" -ge $((100*1024*1024)) ]'

if [[ "$MODE" == "--login" ]]; then
  check "CPU test suite" python3 -m pytest -q tests
else
  check "inside a Slurm job" test -n "${SLURM_JOB_ID:-}"
  check "nvidia-smi" nvidia-smi
  check "torch sees CUDA" python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
  check "GPU is a full GPU with >= 80 GB (not a MIG slice)" python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 2**30
print(p.name, round(gb, 1), 'GiB')
assert gb >= 75, 'GPU memory below 80 GB: a MIG slice or small GPU cannot run the paper settings'"
  check "vllm 0.8.2" python3 -c "import vllm; assert vllm.__version__.startswith('0.8.2'), vllm.__version__"
  check "flash-attn import" python3 -c "import flash_attn"
  check "ALFWorld reset/step on the compute node" python3 week1/check_alfworld.py
  check "model loads offline" python3 -c "
import os
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(os.environ['MODEL_PATH']); print('ok')"
fi

{
  echo "--- resources"
  echo "cpus_visible=$(nproc)"
  echo "slurm_mem_per_node_mb=${SLURM_MEM_PER_NODE:-unset} slurm_cpus=${SLURM_CPUS_ON_NODE:-unset}"
  free -g 2>/dev/null | head -2
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
  python3 -c "import torch,vllm,transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__,'transformers',transformers.__version__)" 2>/dev/null
} >> "$REPORT"

if [[ $fail -ne 0 ]]; then
  echo "PREFLIGHT FAILED ($MODE). See $REPORT" >&2
  exit 1
fi
echo "PREFLIGHT PASSED ($MODE). Report: $REPORT"
