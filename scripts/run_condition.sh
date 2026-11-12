#!/usr/bin/env bash
# Launch one training condition on ALFWorld with Qwen2.5-1.5B-Instruct.
#
#   bash scripts/run_condition.sh <condition> [extra hydra overrides...]
#
# Conditions
#   gigpo            original GiGPO (control; also produces the anchor-group diagnostic)
#   grpo             GRPO baseline
#   gigpo_threshold  condition (b): micro advantage removed for low-signal groups
#   gigpo_weighted   condition (c): micro advantage scaled by group divergence
#   gigpo_dynamic    optional reference: upstream episode-level dynamic sampling
#                    (note: its upstream script also turns on clip-higher 0.28)
#
# Every condition runs the upstream example script of the pinned commit from the
# patched tree, so all paper settings come from upstream. Only the overrides
# below are appended (Hydra: the last value of a key wins), and they are the same
# for every condition except the method-specific ones.
#
# Environment variables (defaults in brackets)
#   SEED [0]              env.seed; validation tasks use SEED+1000. Pair conditions by seed.
#   N_GPUS [2]            GPUs on this node. TP defaults to min(N_GPUS, 2).
#   TP                    vLLM tensor parallel size.
#   TOTAL_EPOCHS [150]    150 = paper budget (16-row placeholder dataset -> 1 step per epoch).
#   TEST_FREQ [5]         validate every N training steps (paper: 5).
#   SAVE_FREQ [10]        checkpoint every N steps. Rerunning the same command resumes.
#   CKPT_KEEP [2]         checkpoints kept per run (each is roughly 20+ GB with optimizer state).
#   MICRO_BS [32]         per-GPU micro batch for actor/log-prob. Lower it if you run out of
#                         memory: it changes gradient accumulation only, not the algorithm.
#   GPU_MEM_UTIL [0.6]    vLLM memory fraction.
#   MODEL_PATH [Qwen/Qwen2.5-1.5B-Instruct]  local path if pre-downloaded.
#   WANDB [1]             1 = log to W&B as well as console.
#   WANDB_PROJECT [gigpo_rl_project]
#   RUN_ROOT [<project>/runs]
#   EXT_TREE [<project>/work/verl-agent-ext]   built by setup/build_ext_tree.sh
#   MIN_GROUP_SIZE, MIN_RETURN_STD   threshold condition (from analysis/choose_thresholds.py)
#   WEIGHT_TAU                       weighted condition (from analysis/choose_thresholds.py)
#   RUN_TAG                          optional suffix for the run name
set -euo pipefail

CONDITION="${1:?usage: run_condition.sh <gigpo|grpo|gigpo_threshold|gigpo_weighted|gigpo_dynamic> [overrides]}"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_TREE="${EXT_TREE:-$PROJECT_ROOT/work/verl-agent-ext}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
SEED="${SEED:-0}"
N_GPUS="${N_GPUS:-2}"
TP="${TP:-$(( N_GPUS < 2 ? N_GPUS : 2 ))}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-10}"
CKPT_KEEP="${CKPT_KEEP:-2}"
MICRO_BS="${MICRO_BS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gigpo_rl_project}"

if [[ ! -f "$EXT_TREE/gigpo/gigpo_ext.py" ]]; then
  echo "Patched tree not found at $EXT_TREE. Run: bash setup/build_ext_tree.sh" >&2
  exit 1
fi

method_args=()
case "$CONDITION" in
  gigpo)
    script=examples/gigpo_trainer/run_alfworld.sh
    method_args+=(algorithm.gigpo.group_filter=none) ;;
  grpo)
    script=examples/grpo_trainer/run_alfworld.sh ;;
  gigpo_threshold)
    script=examples/gigpo_trainer/run_alfworld.sh
    : "${MIN_GROUP_SIZE:?set MIN_GROUP_SIZE (see analysis/choose_thresholds.py)}"
    : "${MIN_RETURN_STD:?set MIN_RETURN_STD (see analysis/choose_thresholds.py)}"
    method_args+=(algorithm.gigpo.group_filter=threshold
                  algorithm.gigpo.min_group_size="$MIN_GROUP_SIZE"
                  algorithm.gigpo.min_return_std="$MIN_RETURN_STD") ;;
  gigpo_weighted)
    script=examples/gigpo_trainer/run_alfworld.sh
    : "${WEIGHT_TAU:?set WEIGHT_TAU (see analysis/choose_thresholds.py)}"
    method_args+=(algorithm.gigpo.group_filter=weighted
                  algorithm.gigpo.weight_tau="$WEIGHT_TAU") ;;
  gigpo_dynamic)
    script=examples/gigpo_dynamic_trainer/run_alfworld.sh
    method_args+=(algorithm.gigpo.group_filter=none) ;;
  *)
    echo "Unknown condition: $CONDITION" >&2; exit 1 ;;
esac

RUN_NAME="${CONDITION}_seed${SEED}${RUN_TAG:+_$RUN_TAG}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

if [[ "$WANDB" == "1" ]]; then LOGGER="['console','wandb']"; else LOGGER="['console']"; fi

common_args=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  env.seed="$SEED"
  trainer.n_gpus_per_node="$N_GPUS"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BS"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BS"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BS"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.test_freq="$TEST_FREQ"
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep="$CKPT_KEEP"
  trainer.resume_mode=auto
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$RUN_NAME"
  "trainer.logger=$LOGGER"
)
if [[ "$CONDITION" != grpo ]]; then
  common_args+=(algorithm.gigpo.diag_dir="$RUN_DIR/gigpo_diag")
fi

# DRY_RUN=1 prints the resolved command (one argument per line) and exits.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "$script" "${method_args[@]}" "${common_args[@]}" "$@"
  exit 0
fi
mkdir -p "$RUN_DIR"

# Provenance: appended on every (re)launch so resumed runs keep their history.
{
  echo "=== launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "condition=$CONDITION seed=$SEED n_gpus=$N_GPUS tp=$TP micro_bs=$MICRO_BS"
  echo "upstream_commit=$(cat "$PROJECT_ROOT/upstream-verl-agent/UPSTREAM_COMMIT")"
  cat "$EXT_TREE/EXT_BUILD_INFO" 2>/dev/null || echo "EXT_BUILD_INFO missing"
  echo "project_git=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo none)"
  echo "project_dirty=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l)"
  echo "command=bash $script vllm ${method_args[*]} ${common_args[*]} $*"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
} >> "$RUN_DIR/launches.txt"
[[ -f "$RUN_DIR/pip_freeze.txt" ]] || python3 -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

cd "$EXT_TREE"
export PYTHONPATH="$EXT_TREE${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
bash "$script" vllm "${method_args[@]}" "${common_args[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
