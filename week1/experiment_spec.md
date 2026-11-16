> Historical Week 1 specification. Current procedure: [`RUNBOOK_OMNI.md`](../RUNBOOK_OMNI.md); current protocol: [`knowledge_base/experiment_protocol.md`](../knowledge_base/experiment_protocol.md).

# Week 1 experiment specification

## Research question

Can the official GiGPO result on ALFWorld be reproduced, and do step-level anchor-group selection methods improve learning over the same GiGPO implementation? The extension comes after a diagnostic of the groups actually formed during training.

## Code revision

- Official repository: `https://github.com/langfengQ/verl-agent`
- Pinned commit: `f974dc5977908d6007ad178c9f3c48f0c7b39331`
- Selection rule: first parent of the 3 June 2025 major update (`f8698a982d731e0968f1859f1d983447fad0017a`). This is the latest mainline commit before that update, not a claim that it exactly matches the authors' private training revision.
- Source snapshot: `upstream-verl-agent/` contains the files from the pinned commit, without the original Git history. `UPSTREAM_COMMIT` records its revision and `UPSTREAM_SHA256SUMS.txt` verifies key source files. Keep this snapshot untouched. Make extension changes in a separate copy or document every difference.

## Baseline configuration from the pinned scripts

| Setting | GiGPO and GRPO |
| --- | --- |
| Task | `alfworld/AlfredTWEnv` (text ALFWorld) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Train / validation data sizes | 16 / 128 placeholder rows; actual tasks come from ALFWorld resets |
| Rollouts per initial state | 8 |
| Environment seed | 0 |
| Episode limit | 50 steps |
| Learning rate | `1e-6` |
| PPO mini batch / micro batch per GPU | 256 / 32 |
| KL loss coefficient | 0.01 |
| Invalid-action penalty | 0.1 |
| Validation sampling | temperature 0.4, `do_sample=True` |
| Validation frequency | every 5 training steps, plus the final step |
| Total epochs | 150 |
| Checkpoint frequency | `-1` in the supplied scripts; our launcher sets `SAVE_FREQ=5`, keep 2, auto-resume, for every condition |

GiGPO additionally sets `algorithm.gamma=0.95`, `algorithm.gigpo.step_advantage_w=1.0`, and `algorithm.gigpo.mode=mean_std_norm`. These define the method rather than a mismatch to remove from GRPO. Both scripts request two GPUs and tensor model parallel size two. On Omni each run uses one full GPU (`N_GPUS=1`, tensor parallel 1) and the hardware knobs in `cluster/omni/resources.env`, identically for all methods; see `RUNBOOK_OMNI.md`.

## What to compare

For Week 2, run GiGPO and GRPO with the same pinned code, model, ALFWorld split, environment seed, rollout count, update budget, and evaluation settings. Record the exact command, commit, environment/package versions, hardware, elapsed time, and log file for each run. The paper's 86.7% GiGPO success rate is a reference target, not a pass/fail threshold for the one-epoch smoke test.

The validation script samples actions (`temperature=0.4`), so a single reported success rate can vary. Preserve the original evaluation configuration for reproduction; for a stronger comparison, rerun evaluation with recorded seeds or repeated evaluations and report variability. Do not tune the extension using the held-out evaluation results.

## Week 3 diagnostic, prepared now

Implemented. Every GiGPO-family run logs `gigpo_diag/*` and writes `gigpo_diag/step_groups.jsonl`. `analysis/choose_thresholds.py` applies the predeclared threshold rule and the null-result gate to the seed-0 control run. Degenerate groups are defined by computed return signal, not by action identity. Exactly degenerate groups already get zero micro advantage upstream, so the extension targets small or near-zero-spread groups. Episode-level dynamic sampling exists in the pinned snapshot (`examples/gigpo_dynamic_trainer`) and is a different axis. See `knowledge_base/extension_design.md`.

## Completion gate for Week 1

- The pinned commit is present and unchanged.
- The baseline comparison check passes.
- ALFWorld can reset and accept actions on the Linux machine.
- `bash week1/smoke_gigpo.sh` prints SMOKE PASSED: updates, validation, diagnostics, checkpoint and resume all verified, with command, versions and log captured.

The last two items require the separate Linux training machine and remain pending until run there.
