#!/usr/bin/env bash
# Launch one training condition on ALFWorld with Qwen2.5-1.5B-Instruct.
#
#   bash scripts/run_condition.sh <condition> [extra hydra overrides...]
#
# On Omni this script is called from cluster/omni/job.sbatch inside a GPU job.
# Do not call it on the login node except with DRY_RUN=1.
#
# Conditions
#   gigpo            original GiGPO (control; also produces the anchor-group diagnostic)
##   gigpo_threshold  condition (b): micro advantage removed for low-signal groups
#   gigpo_weighted   condition (c): micro advantage scaled by group divergence
#
# The three seed-0 conditions run the upstream GiGPO example script of the pinned commit from the
# patched tree, so all paper settings come from upstream. Only the overrides
# below are appended (Hydra: the last value of a key wins). They are identical
# for every condition except the method-specific ones.
#
# Environment variables (defaults in brackets). Only the HARDWARE knobs may be
# changed to make a run fit; see RUNBOOK_OMNI.md, "Fitting a run".
#   SEED [0]              fixed to 0 for this handoff; validation tasks use 1000.
#   TOTAL_EPOCHS [150]    training steps (16-row placeholder dataset -> 1 step per epoch).
#                         150 = paper budget. Change only with written human approval.
#   TEST_FREQ [5]         validate every N training steps (paper: 5).
#   SAVE_FREQ [5]         checkpoint every N steps. Relaunching the same command resumes.
#   CKPT_KEEP [2]         checkpoints kept per run (each is roughly 20 GB with optimizer state).
#   HARDWARE knobs (no effect on the algorithm):
#   N_GPUS [1]            GPUs used by this run. TP defaults to min(N_GPUS, 2).
#   TP                    vLLM tensor parallel size.
#   MICRO_BS [16]         per-GPU micro batch for actor update and log-prob passes.
#   GPU_MEM_UTIL [0.5]    vLLM share of GPU memory.
#   VAL_BATCH [128]       validation environments run at once (always 128 tasks in total).
#   REF_OFFLOAD [True]    keep the reference model on CPU (upstream default) or GPU (False).
#   Other:
#   MODEL_PATH            local model snapshot (set by cluster/omni/env.sh).
#   WANDB [0]             1 = also log to W&B (Omni compute nodes are offline: keep 0).
#   RUN_ROOT [<project>/runs]
#   EXT_TREE [<project>/work/verl-agent-ext]   built by setup/build_ext_tree.sh
#   MIN_GROUP_SIZE, MIN_RETURN_STD   threshold condition (results/diagnostic/thresholds.env)
#   WEIGHT_TAU                       weighted condition (results/diagnostic/thresholds.env)
#   RUN_TAG                          optional suffix for the run name (e.g. smoke)
set -euo pipefail

CONDITION="${1:?usage: run_condition.sh <gigpo|gigpo_threshold|gigpo_weighted> [overrides]}"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_TREE="${EXT_TREE:-$PROJECT_ROOT/work/verl-agent-ext}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
SEED="${SEED:-0}"
if [[ "$SEED" != "0" ]]; then
  echo "Only SEED=0 is allowed by the three-experiment handoff (got SEED=$SEED)." >&2
  exit 2
fi
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
CKPT_KEEP="${CKPT_KEEP:-2}"
N_GPUS="${N_GPUS:-1}"
TP="${TP:-$(( N_GPUS < 2 ? N_GPUS : 2 ))}"
MICRO_BS="${MICRO_BS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
VAL_BATCH="${VAL_BATCH:-128}"
REF_OFFLOAD="${REF_OFFLOAD:-True}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-gigpo_rl_project}"

if [[ ! -f "$EXT_TREE/gigpo/gigpo_ext.py" ]]; then
  echo "Patched tree not found at $EXT_TREE. Run: bash setup/build_ext_tree.sh" >&2
  exit 1
fi

# Settings that define the experiment. They come from the upstream scripts and
# must not be overridden on the command line; a mismatch would make the
# conditions incomparable with each other and with the paper.
PROTECTED=(
  data.train_batch_size env.seed env.rollout.n env.max_steps env.env_name
  actor_rollout_ref.actor.ppo_mini_batch_size actor_rollout_ref.actor.optim.lr
  actor_rollout_ref.actor.kl_loss_coef actor_rollout_ref.actor.use_kl_loss
  actor_rollout_ref.actor.invalid_action_penalty_coef
  actor_rollout_ref.rollout.val_kwargs.temperature actor_rollout_ref.rollout.val_kwargs.do_sample
  data.max_prompt_length data.max_response_length
  algorithm.gamma algorithm.gigpo.step_advantage_w algorithm.gigpo.mode algorithm.adv_estimator
  algorithm.gigpo.group_filter trainer.total_epochs trainer.total_training_steps
)
for arg in "$@"; do
  key="${arg%%=*}"; key="${key#+}"
  for p in "${PROTECTED[@]}"; do
    if [[ "$key" == "$p" ]]; then
      echo "Refusing override '$arg': $p is a protected experiment setting." >&2
      echo "Use the environment variables documented at the top of this script, or ask a human." >&2
      exit 2
    fi
  done
done

method_args=()
case "$CONDITION" in
  gigpo)
    script=examples/gigpo_trainer/run_alfworld.sh
    method_args+=(algorithm.gigpo.group_filter=none) ;;
  gigpo_threshold)
    script=examples/gigpo_trainer/run_alfworld.sh
    : "${MIN_GROUP_SIZE:?set MIN_GROUP_SIZE (source results/diagnostic/thresholds.env)}"
    : "${MIN_RETURN_STD:?set MIN_RETURN_STD (source results/diagnostic/thresholds.env)}"
    method_args+=(algorithm.gigpo.group_filter=threshold
                  algorithm.gigpo.min_group_size="$MIN_GROUP_SIZE"
                  algorithm.gigpo.min_return_std="$MIN_RETURN_STD") ;;
  gigpo_weighted)
    script=examples/gigpo_trainer/run_alfworld.sh
    : "${WEIGHT_TAU:?set WEIGHT_TAU (source results/diagnostic/thresholds.env)}"
    method_args+=(algorithm.gigpo.group_filter=weighted
                  algorithm.gigpo.weight_tau="$WEIGHT_TAU") ;;
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
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_OFFLOAD"
  data.val_batch_size="$VAL_BATCH"
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
common_args+=(algorithm.gigpo.diag_dir="$RUN_DIR/gigpo_diag")

# DRY_RUN=1 prints the resolved command (one argument per line) and exits.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "$script" "${method_args[@]}" "${common_args[@]}" "$@"
  exit 0
fi
mkdir -p "$RUN_DIR"

# Provenance: appended on every (re)launch so resumed runs keep their history.
{
  echo "=== launch $(date -u +%Y-%m-%dT%H:%M:%SZ) slurm_job=${SLURM_JOB_ID:-none} node=$(hostname)"
  echo "condition=$CONDITION seed=$SEED total_epochs=$TOTAL_EPOCHS save_freq=$SAVE_FREQ"
  echo "hardware: n_gpus=$N_GPUS tp=$TP micro_bs=$MICRO_BS gpu_mem_util=$GPU_MEM_UTIL val_batch=$VAL_BATCH ref_offload=$REF_OFFLOAD"
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
