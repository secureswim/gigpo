#!/usr/bin/env bash
# Single-MIG smoke profile; the paper baseline scripts remain unchanged.
set -euo pipefail
ROOT=/home/vansh23580/rl-project/gigpo
source /home/vansh23580/miniconda3/etc/profile.d/conda.sh
conda activate /home/vansh23580/rl-project/envs/gigpo
export HF_HOME="$ROOT/cache/huggingface"
export ALFWORLD_DATA="$ROOT/cache/alfworld"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 VLLM_USE_V1=0
export RAY_TMPDIR="/tmp/gigpo-${SLURM_JOB_ID:?}"
mkdir -p "$RAY_TMPDIR" "$ROOT/runs/omni-$SLURM_JOB_ID"
RUN="$ROOT/runs/omni-$SLURM_JOB_ID"
python -m pip freeze > "$RUN/packages.txt"
python "$ROOT/week1/check_alfworld.py"
cd "$ROOT/upstream-verl-agent"
# Data preparation already ran on the internet-connected master.
# Strip just that command from a temporary copy of the unchanged baseline.
sed -e '/^ENGINE=/a shift' \
  -e '/^python3 -m examples.data_preprocess.prepare/,/--val_data_size \$val_data_size/d' \
  examples/gigpo_trainer/run_alfworld.sh > "$RUN/baseline-offline.sh"
bash "$RUN/baseline-offline.sh" vllm \
  trainer.n_gpus_per_node=1 \
  data.train_batch_size=4 data.val_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.test_freq=1 trainer.val_before_train=False \
  "trainer.logger=['console']" trainer.experiment_name=omni_mig_smoke \
  trainer.default_local_dir="$RUN/checkpoints" \
  "$@" \
  2>&1 | tee "$RUN/train.log"
