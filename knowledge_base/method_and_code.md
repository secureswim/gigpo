# GiGPO method and code map

## Problem

An ALFWorld agent gets task feedback only at the end of a multi-step episode, so a trajectory-level signal does not say which earlier action helped or hurt. GRPO compares complete rollouts that start from the same initial state. GiGPO adds a step-level comparison among the actions taken when different rollouts reach the same anchor observation.

## Advantage calculation in the pinned revision

`gigpo/core_gigpo.py::compute_gigpo_outcome_advantage` computes the episode advantage, builds anchor-state groups (exact match of `anchor_obs` within the same episode group `uid`), computes a normalised step advantage within each group, and returns `episode + step_advantage_w * step`. The ALFWorld script sets ω = 1.0 and `mean_std_norm`, which is the paper's "w/ std" variant (86.7%). The trainer computes discounted step returns with γ = 0.95 before the advantage call (`ray_trainer.py`, `fit`). Invalid actions subtract 0.1 from that step's return.

A group of size 1, or one with equal returns, has zero micro advantage. With std normalisation the micro advantage is scale-free. See `extension_design.md`.

## Upstream files

| File | Role |
| --- | --- |
| `examples/gigpo_trainer/run_alfworld.sh`, `examples/grpo_trainer/run_alfworld.sh` | paper settings; our run scripts call these and append overrides |
| `examples/gigpo_dynamic_trainer/run_alfworld.sh` | GiGPO with DAPO dynamic sampling and clip-higher (in the snapshot) |
| `gigpo/core_gigpo.py` | episode and step advantage, anchor grouping, normalisation |
| `verl/trainer/ppo/ray_trainer.py` | rollout, reward, advantage (driver side), validation, checkpointing, training loop |
| `agent_system/multi_turn_rollout/rollout_loop.py` | multi-turn rollout; dynamic sampling loop |
| `agent_system/environments/env_manager.py` | env creation (train seed = `env.seed`, validation seed = `env.seed + 1000`) and `success_rate` |
| `agent_system/environments/env_package/alfworld/envs.py` | text ALFWorld wrapper; reward `10 * won` |
| `agent_system/environments/env_package/alfworld/configs/config_tw.yaml` | data paths; validation uses `valid_seen` |
| `examples/data_preprocess/prepare.py` | placeholder parquet rows (16 train / 128 val); real tasks come from env resets |

## Our extension (applied to a copy, `work/verl-agent-ext/`)

| File | Change |
| --- | --- |
| `extension/gigpo_ext.py` → `gigpo/gigpo_ext.py` | diagnostics, `threshold` and `weighted` group selection |
| `extension/verl-agent-ext.patch` | `ray_trainer.py`: call the extension for GiGPO, log `gigpo_diag/*`, dump per-group arrays, log `env/steps_collected`. `rollout_loop.py`: count environment steps, including rollouts discarded by dynamic sampling. `dp_actor.py`: log `actor/entropy` (logging only). `ppo_trainer.yaml`: new `algorithm.gigpo.*` keys with defaults that reproduce upstream. The three `examples/*/run_alfworld.sh` scripts forward `"${@:2}"` instead of `$@`, so the engine name is not passed to Hydra as an override, and they skip the data-preparation step (which downloads from the internet) when `~/data/verl-agent/text/{train,test}.parquet` already exist. |

`setup/build_ext_tree.sh` builds the copy; on Omni, `cluster/omni/prepare.sh` builds it and installs it editable, so `import verl` resolves to the patched code (preflight checks this). Rebuild only by rerunning `prepare.sh`. `UPSTREAM_SHA256SUMS.txt` also covers every file the patch touches, so a patched file cannot drift silently.

## Logged metric names (verified in the pinned code)

- Validation: `val/success_rate` (plus per-task `val/<task>_success_rate`) and `val/test_score/alfworld/AlfredTWEnv`.
- Training: `episode/reward/{mean,max,min}`, `episode/length/{mean,max,…}`, `actor/{pg_loss,pg_clipfrac,ppo_kl,kl_loss,grad_norm}`, `timing_s/*`.
- Added by us: `actor/entropy`, `env/steps_collected`, `gigpo_diag/*`.
- Console format: `step:N - key:value - …`, rounded to 3 decimals. W&B keeps full precision.

## References

- Feng et al., [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978), NeurIPS 2025.
- [Official implementation](https://github.com/langfengQ/verl-agent), pinned commit in `upstream-verl-agent/UPSTREAM_COMMIT`.
