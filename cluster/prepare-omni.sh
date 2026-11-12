#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vansh23580/rl-project/gigpo
ENV=/home/vansh23580/rl-project/envs/gigpo
source /home/vansh23580/miniconda3/etc/profile.d/conda.sh
if [[ ! -x $ENV/bin/python ]]; then
  conda create -y -p "$ENV" python=3.12
fi
conda activate "$ENV"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export PIP_CONSTRAINT="$ROOT/cluster/constraints.txt"
export TMPDIR=/home/vansh23580/rl-project/gigpo/cache/tmp
mkdir -p "$TMPDIR"
if ! python -c 'import importlib.metadata as m; assert m.version("xgrammar") == "0.1.16"'; then
  python -m pip install 'xgrammar @ git+https://github.com/mlc-ai/xgrammar.git@989222175c2a30fb7987d8bcce35bec1bf6817f2'
fi
python -m pip install 'torch==2.6.0' 'vllm==0.8.2' 'transformers==4.51.1' 'peft==0.15.1' 'accelerate==1.6.0' 'datasets==3.5.0' 'ray[default]==2.43.0' 'tensordict==0.6.2' 'torchdata==0.11.0' 'numpy<2' 'setuptools<81' packaging ninja wheel
(
  set +u
  source /etc/profile.d/modules.sh
  module load cuda12.8/toolkit/12.8.1
  export TORCH_CUDA_ARCH_LIST=9.0
  python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
)
python -m pip install -e "$ROOT/upstream-verl-agent" 'gymnasium==0.29.1' 'stable-baselines3==2.6.0' 'alfworld==0.4.2' 'thinc==8.2.5' 'spacy==3.7.5' 'numpy==1.26.4' 'contourpy==1.3.2'
export HF_HOME="$ROOT/cache/huggingface"
export ALFWORLD_DATA="$ROOT/cache/alfworld"
mkdir -p "$ALFWORLD_DATA"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
print('MODEL_SNAPSHOT', path)
PY
if [[ ! -f $ROOT/cluster/alfworld-download-complete ]]; then
  alfworld-download -f
  touch "$ROOT/cluster/alfworld-download-complete"
fi
cd "$ROOT"
python week1/check_baseline_scripts.py
python week1/check_alfworld.py
cd upstream-verl-agent
python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128
python -m pip freeze > "$ROOT/cluster/omni-packages.txt"
touch "$ROOT/cluster/preparation-complete"
