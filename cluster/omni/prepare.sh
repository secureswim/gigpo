#!/usr/bin/env bash
# One-time preparation on the Omni LOGIN node (needs internet, no GPU).
#
#   cd <repo root> && bash cluster/omni/prepare.sh
#
# Based on the preparation that produced the passing smoke run (Omni job 10333),
# generalised to any account and changed to install the PATCHED tree
# (work/verl-agent-ext) instead of the pristine snapshot. Safe to re-run: each
# step is skipped or repeated idempotently. Ends with "PREPARE OK".
#
# Version notes (why this differs from the upstream README's install list):
#   * vLLM 0.8.2 needs xgrammar 0.1.16, whose wheel is no longer on the package
#     index; it is built from the official tag (commit 9892221...).
#   * The README's thinc 8.3.4 pulls NumPy 2, which breaks vLLM 0.8.2. NumPy
#     1.26.4 with thinc 8.2.5 / spaCy 3.7.5 and the pins in constraints.txt work.
#   * FlashAttention is installed from its official prebuilt wheel, not compiled.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# shellcheck disable=SC1091
source cluster/omni/env.sh
ROOT="$GIGPO_ROOT"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run prepare.sh on the login node, not inside a Slurm job (compute nodes are offline)." >&2
  exit 1
fi
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || { echo "Miniconda not found at $CONDA_BASE" >&2; exit 1; }

mkdir -p "$ROOT/cache/tmp" "$HF_HOME" "$ALFWORLD_DATA" "$ROOT/slurm-logs" "$ROOT/runs"
export TMPDIR="$ROOT/cache/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export PIP_CONSTRAINT="$ROOT/cluster/omni/constraints.txt"

# 1. Python environment
if [[ ! -x "$GIGPO_ENV/bin/python" ]]; then
  conda create -y -p "$GIGPO_ENV" python=3.12
fi
conda activate "$GIGPO_ENV"

# 2. Pinned packages
if ! python -c 'import importlib.metadata as m; assert m.version("xgrammar") == "0.1.16"' 2>/dev/null; then
  python -m pip install 'xgrammar @ git+https://github.com/mlc-ai/xgrammar.git@989222175c2a30fb7987d8bcce35bec1bf6817f2'
fi
python -m pip install 'torch==2.6.0' 'vllm==0.8.2' 'transformers==4.51.1' 'peft==0.15.1' \
  'accelerate==1.6.0' 'datasets==3.5.0' 'ray[default]==2.43.0' 'tensordict==0.6.2' \
  'torchdata==0.11.0' 'numpy<2' 'setuptools<81' packaging ninja wheel
if ! python -c 'import flash_attn' 2>/dev/null; then
  (
    set +u
    if [[ -f /etc/profile.d/modules.sh ]]; then
      source /etc/profile.d/modules.sh
      module load cuda12.8/toolkit/12.8.1 || true
    fi
    export TORCH_CUDA_ARCH_LIST=9.0
    python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
  )
fi

# 3. Patched training tree, installed editable so `import verl` resolves to it
bash setup/build_ext_tree.sh
python -m pip install -e "$EXT_TREE" 'gymnasium==0.29.1' 'stable-baselines3==2.6.0' 'alfworld==0.4.2' \
  'thinc==8.2.5' 'spacy==3.7.5' 'numpy==1.26.4' 'contourpy==1.3.2' 'pytest' 'matplotlib'
python - <<'PY'
import os, verl
p = os.path.dirname(verl.__file__)
assert "verl-agent-ext" in p, f"verl resolves to {p}; expected the patched tree work/verl-agent-ext"
print("verl package:", p)
PY

# 4. Model (pinned revision) and ALFWorld data
python - <<PY
from huggingface_hub import snapshot_download
print("model:", snapshot_download("$MODEL_ID", revision="$MODEL_REVISION"))
PY
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Model snapshot missing at $MODEL_PATH" >&2; exit 1; }
if [[ ! -f "$ALFWORLD_DATA/.download-complete" ]]; then
  alfworld-download -f
  touch "$ALFWORLD_DATA/.download-complete"
fi

# 5. Placeholder parquet files (the launch scripts skip this step on compute nodes)
(cd "$EXT_TREE" && python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128)

# 6. Checks that need no GPU, and the environment lock
bash setup/preflight.sh --login
python -m pip freeze > cluster/omni/env_lock.txt
echo "PREPARE OK"
