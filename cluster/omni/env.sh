# Environment for every Omni command. Source it from the repository root:
#
#   source cluster/omni/env.sh
#
# Works on the login node (internet) and inside Slurm jobs (compute nodes are
# offline). Every path is relative to this repository or to $HOME, so it works
# for any account. Override a variable by exporting it before sourcing.

GIGPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GIGPO_ROOT
export CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
export GIGPO_ENV="${GIGPO_ENV:-$GIGPO_ROOT/cache/env}"
export HF_HOME="${HF_HOME:-$GIGPO_ROOT/cache/huggingface}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$GIGPO_ROOT/cache/alfworld}"
export EXT_TREE="${EXT_TREE:-$GIGPO_ROOT/work/verl-agent-ext}"
export RUN_ROOT="${RUN_ROOT:-$GIGPO_ROOT/runs}"

# The exact model revision used by the passing smoke run (Omni job 10333).
export MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
export MODEL_REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
export MODEL_PATH="${MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/$MODEL_REVISION}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export VLLM_USE_V1=0           # verl at the pinned commit uses the vLLM V0 engine
export WANDB="${WANDB:-0}"     # console logging only; the analysis reads train.log

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Compute node: no internet. Everything must already be cached by prepare.sh.
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
  export RAY_TMPDIR="/tmp/gigpo-$SLURM_JOB_ID"   # short path: Ray socket paths are length-limited
  mkdir -p "$RAY_TMPDIR"
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if [[ -x "$GIGPO_ENV/bin/python" ]]; then
    conda activate "$GIGPO_ENV"
  fi
else
  echo "env.sh: conda not found at $CONDA_BASE (set CONDA_BASE)" >&2
fi
