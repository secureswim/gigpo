#!/usr/bin/env bash
# One-shot environment setup on the Linux GPU machine (run from the project root).
#
#   bash setup/install.sh
#
# Steps (each is skipped if already done):
#   1. conda env "verl-agent-gigpo" (Python 3.12), packages pinned as in the
#      upstream README at the pinned commit (torch 2.6.0/cu124, vllm 0.8.2, ...)
#   2. build the patched tree work/verl-agent-ext and install it editable
#   3. download ALFWorld data and the Qwen2.5-1.5B-Instruct weights
#   4. save the exact package set to setup/env_lock.txt (commit it so the team
#      and every run use the same versions)
#
# Variables: ENV_NAME [verl-agent-gigpo], CUDA_WHL [cu124], MODEL_DIR [$HOME/models],
#            ALFWORLD_DATA [$HOME/.cache/alfworld], SKIP_MODEL=1 to skip the model download.
# It needs internet. It does not need a GPU, except flash-attn, which compiles
# against CUDA and can take a long time if no prebuilt wheel matches.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-verl-agent-gigpo}"
CUDA_WHL="${CUDA_WHL:-cu124}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"

command -v conda >/dev/null || { echo "conda not found. Install Miniconda first." >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.6.0') else 1)" 2>/dev/null || \
  python -m pip install torch==2.6.0 --index-url "https://download.pytorch.org/whl/$CUDA_WHL"
python -c "import flash_attn" 2>/dev/null || python -m pip install flash-attn --no-build-isolation

bash "$PROJECT_ROOT/setup/build_ext_tree.sh"
python -m pip install -e "$PROJECT_ROOT/work/verl-agent-ext"
python -m pip install vllm==0.8.2
python -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld thinc==8.3.4
python -m pip install pytest wandb matplotlib

# The editable install above points 'verl' at the patched tree; check it.
python - <<'PY'
import verl, os
p = os.path.dirname(verl.__file__)
assert "verl-agent-ext" in p, f"verl resolves to {p}, expected work/verl-agent-ext"
print("verl package:", p)
PY

mkdir -p "$ALFWORLD_DATA"
if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  alfworld-download -f
fi

if [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  mkdir -p "$MODEL_DIR"
  python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="$MODEL_DIR/Qwen2.5-1.5B-Instruct")
print("model:", p)
PY
fi

python -m pip freeze > "$PROJECT_ROOT/setup/env_lock.txt"
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)  $(python --version)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
} >> "$PROJECT_ROOT/setup/env_lock.txt"

cat <<EOF

Install finished. Add these to your shell profile (or source them before each run):
  conda activate $ENV_NAME
  export ALFWORLD_DATA=$ALFWORLD_DATA
  export MODEL_PATH=$MODEL_DIR/Qwen2.5-1.5B-Instruct
Then run: bash setup/preflight.sh
EOF
