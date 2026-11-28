#!/bin/bash
set -eo pipefail
test "$SLURM_JOB_ID" = 11018
cd ~/rl-project/gigpo-fb72287
source operator/env-b200.sh
r=$HOME/rl-project/runs
export EVAL_ROOT=$r/weighted-reevaluation-11018
export EXT_TREE=$EVAL_ROOT/tree
export PYTHONPATH=$EXT_TREE:$PYTHONPATH
export PYTHONHASHSEED=0
cd "$EXT_TREE"
exec 9>"$EVAL_ROOT/evaluate.lock"
flock -n 9
ckpt=$r/extensions-10605/gigpo_weighted_seed0/checkpoints/global_step_150
test -s "$ckpt/actor/model_world_size_1_rank_0.pt"
for seed in 1 2 3 4 5; do
 dir=$EVAL_ROOT/weighted-$seed
 mkdir -p "$dir"
 if test -f "$dir/ok"; then continue; fi
 export EVAL_EPISODES=$dir/episodes.jsonl
 : > "$EVAL_EPISODES"
 echo "$(date -Is) START weighted $seed"
 bash examples/gigpo_trainer/run_alfworld.sh vllm actor_rollout_ref.model.path="$MODEL_PATH" env.seed="$((seed * 128))" ++actor_rollout_ref.rollout.seed="$seed" trainer.n_gpus_per_node=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 "trainer.logger=['console']" trainer.default_local_dir="$dir" trainer.resume_mode=resume_path trainer.resume_from_path="$ckpt" trainer.del_local_ckpt_after_load=False trainer.val_before_train=True ++trainer.val_only=True > "$dir/eval.log" 2>&1
 python3 "$EVAL_ROOT/verify.py" "$seed"
 touch "$dir/ok"
 echo "$(date -Is) DONE weighted $seed"
done
echo "$(date -Is) COMPLETE"
