# Baseline and evaluation protocol

The operational steps (install, resources, smoke, budget, launches, monitoring) are in [`RUNBOOK_OMNI.md`](../RUNBOOK_OMNI.md). This file defines **what** is compared and measured.

## Conditions and matching

All conditions are launched by `cluster/omni/job.sbatch <condition>`, which calls `scripts/run_condition.sh`. That script runs the pinned upstream example script from the patched tree and appends identical shared overrides for every condition. `tests/test_launch_compose.py` composes the full configuration of each condition and checks this.

| Condition | What differs |
| --- | --- |
| `gigpo` | seed-0 control: `group_filter=none` (bit-identical advantages to upstream) |
| `gigpo_threshold` | seed-0 extension: `group_filter=threshold` with `MIN_GROUP_SIZE` and `MIN_RETURN_STD` from the diagnostic |
| `gigpo_weighted` | seed-0 extension: `group_filter=weighted` with `WEIGHT_TAU` from the diagnostic |

Only these three seed-0 conditions may be launched under this handoff. Both extensions run when thresholds can be derived, even if the threshold diagnostic predicts a small effect. If no groups have return spread by the control's final step, no thresholds are derived and the runbook requests a new plan. GRPO and dynamic GiGPO remain in the pinned upstream snapshot for reference but are not run.

GiGPO keeps its method-specific settings: γ = 0.95, ω = 1, `mean_std_norm`.

**Protected settings** must never be overridden; `run_condition.sh` refuses them:
- tasks per step (16) and rollouts per task (8)
- PPO minibatch (256) and learning rate
- KL settings and the invalid-action penalty
- prompt and response lengths, and the episode limit (50)
- validation sampling
- γ, ω, normalisation mode, estimator, filter type
- the number of training steps

**Hardware knobs** (GPU count, micro-batch, vLLM memory share, validation batch, reference-model offload) change memory and speed only. They are fixed once in `cluster/omni/resources.env` and are identical for all runs.

Paper reference (Table 1, Qwen2.5-1.5B, mean ± std over 3 seeds): GiGPO w/ std (our `mean_std_norm` setting) **86.7 ± 1.7**, GRPO **72.8 ± 3.6**. The paper trains 150 iterations on 2×H100.

## Fixed choices (decided before any result)

- **Checkpoints:** `SAVE_FREQ=5`, keep 2, `resume_mode=auto`, identical for every condition. Runs longer than the Slurm time limit are submitted as job chains (`cluster/omni/submit_chain.sh`). Each job resumes from the latest checkpoint. After a resume the environment RNG restarts, so the task sequence differs from an uninterrupted run. Note any resume in the run record.
- **Success threshold for "steps to target": 0.70 validation success.** It is a fixed reference threshold for comparing the three GiGPO-family conditions. Change it only before the first full run, and record the change here.
- **Seed:** only `env.seed=0` is authorized in this handoff; validation uses seed 1000. All three conditions are paired on this seed. vLLM sampling is not seeded, so a rerun is not bit-identical. One-seed findings are preliminary.
- **Budget:** 150 steps per run, as in the paper. A shorter budget (`TOTAL_EPOCHS`) is allowed only by H2 approval, and must then be the same for every compared condition.
- **Validation:** the pinned scripts sample at temperature 0.4 on 128 tasks drawn from `valid_seen` (`eval_in_distribution`) every 5 steps. `VAL_BATCH` only sets how many run at once. Each validation round resets the environments, so successive points appear to use different task draws. Points are noisy. Report the mean of the last 3 validation points next to the final one.
- **Extension thresholds:** set by the rule in `extension_design.md` from the seed-0 GiGPO run's training batches.

## Outcome measures (all produced by `analysis/summarize.py` and `analysis/plot_curves.py`)

- **Primary:** `val/success_rate`, final and mean of the last 3 points.
- **Learning curve:** success vs. training step and vs. cumulative `env/steps_collected`.
- **Efficiency:** training steps and environment steps to 0.70. Never reached → NaN, reported as such.
- **Stability:** `actor/grad_norm` (std, max), `actor/entropy` (first vs. last third), training reward std, episode length.
- **Mechanism:** `gigpo_diag/*`, i.e. what each filter removed.

## Result record

Copy `results/TEMPLATE.md` to `results/<run>.md` for each run. Keep checkpoints and logs outside Git (`runs/`, `slurm-logs/`), and put their paths in the record. Record failed runs and resumes too. Approvals go in `results/decisions.md`.
