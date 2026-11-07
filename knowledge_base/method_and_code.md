# GiGPO method and code map

## Problem

An ALFWorld agent receives task feedback after a multi-step episode. A trajectory-level signal does not identify which earlier action helped or hurt. GRPO compares complete rollouts from a common initial state. GiGPO adds a step-level comparison among actions taken when different rollouts encounter the same anchor observation.

## Advantage calculation in this source revision

`gigpo/core_gigpo.py` contains `compute_gigpo_outcome_advantage`. It computes an episode advantage, constructs anchor-state groups, computes a normalized step advantage within each group, and adds the two using `algorithm.gigpo.step_advantage_w`. The pinned ALFWorld script sets that weight to `1.0` and uses `mean_std_norm`. The trainer computes discounted step returns with `algorithm.gamma=0.95` before calling the advantage function.

The step group is formed from repeated anchor observations **within the same initial-state group**. The proposal's extension changes what to do with a group after formation; it should not silently change the state-matching rule. Group size one and groups with equal discounted returns have zero centered step signal. Identical actions by themselves do not prove zero return variance.

## Code paths to inspect

| File | What it does |
| --- | --- |
| `upstream-verl-agent/examples/gigpo_trainer/run_alfworld.sh` | GiGPO model, environment, rollout, optimizer, validation, and logging settings |
| `upstream-verl-agent/examples/grpo_trainer/run_alfworld.sh` | GRPO baseline settings |
| `upstream-verl-agent/gigpo/core_gigpo.py` | Episode and step advantage, anchor grouping, normalization |
| `upstream-verl-agent/verl/trainer/ppo/ray_trainer.py` | Rollout, reward, advantage, validation, and training loop |
| `upstream-verl-agent/agent_system/environments/env_manager.py` | ALFWorld environment creation and success-rate collection |
| `upstream-verl-agent/agent_system/environments/env_package/alfworld/envs.py` | Text ALFWorld wrapper, reset, actions, reward |
| `upstream-verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml` | ALFWorld text environment and dataset paths |
| `upstream-verl-agent/examples/data_preprocess/prepare.py` | Creates placeholder parquet rows; the actual tasks come from environment resets |

## Evaluation interpretation

The ALFWorld wrapper gives a text-task terminal reward of `10 * won`. The environment manager records `success_rate` from the success flag, and the trainer averages validation success metrics. Read the exact logged metric name from the first successful validation run before writing any analysis script. The prepared dataset size is not the number of distinct ALFWorld tasks.

## References

- Feng et al., [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978), NeurIPS 2025.
- [Official implementation](https://github.com/langfengQ/verl-agent), source commit recorded in `upstream-verl-agent/UPSTREAM_COMMIT`.
