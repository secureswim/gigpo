# Baseline and evaluation protocol

## Gate 1: environment

Run `bash setup/install.sh`, then `bash setup/preflight.sh`. All checks must pass. `runs/preflight.txt` records the versions. Commit `setup/env_lock.txt` so every teammate and every run uses the same package set.

## Gate 2: smoke run

Run `bash week1/smoke_gigpo.sh`. It trains one GiGPO step with validation and a checkpoint, then relaunches with a 2-step budget, which must resume from the step-1 checkpoint. `week1/check_smoke.py` must print `SMOKE PASSED`. Record `timing_s/step` in a run record: it sets the compute budget (see `compute_and_team.md`). Do not compare the smoke score with the paper.

## Gate 3: matched baselines

```bash
SEED=0 bash scripts/run_condition.sh gigpo
SEED=0 bash scripts/run_condition.sh grpo
```

Both use the pinned upstream example scripts from the patched tree, with identical shared overrides (`tests/test_gigpo_ext.py::test_conditions_differ_only_in_method_settings` checks this). GiGPO keeps its method-specific settings: γ = 0.95, ω = 1, `mean_std_norm`.

Paper reference (Table 1, Qwen2.5-1.5B, mean ± std over 3 seeds): GiGPO w/ std (our `mean_std_norm` setting) **86.7 ± 1.7**, GRPO **72.8 ± 3.6**. The paper trains 150 iterations on 2×H100.

## Fixed choices (decided before any result)

- **Checkpoints:** `SAVE_FREQ=10`, keep 2, `resume_mode=auto`, identical for every condition. Rerunning the same command resumes. After a resume the environment RNG restarts, so the task sequence differs from an uninterrupted run. Note any resume in the run record.
- **Success threshold for "steps to target": 0.70 validation success.** It is reachable by both baselines in the paper (GRPO final 72.8%), so it measures speed rather than only GiGPO's ceiling. Change it only before the first full run, and record the change here.
- **Seeds:** `env.seed` controls both training tasks and validation tasks (validation uses seed + 1000). Compare conditions **paired by seed**. Minimum plan: seed 0 for all conditions, then seed 1 for GiGPO and any extension condition that looks better. vLLM sampling is not seeded, so a rerun is not bit-identical.
- **Validation:** the pinned scripts sample at temperature 0.4 on 128 environments drawn from `valid_seen` (`eval_in_distribution`) every 5 steps. Each validation round resets the environments, so successive points appear to use different task draws. Points are noisy. Report the mean of the last 3 validation points next to the final one.
- **Extension thresholds:** set by the rule in `extension_design.md` from the seed-0 GiGPO run's training batches.

## Outcome measures (all produced by `analysis/summarize.py` and `analysis/plot_curves.py`)

- **Primary:** `val/success_rate`, final and mean of the last 3 points.
- **Learning curve:** success vs. training step and vs. cumulative `env/steps_collected`.
- **Efficiency:** training steps and environment steps to 0.70. Never reached → NaN, reported as such.
- **Stability:** `actor/grad_norm` (std, max), `actor/entropy` (first vs. last third), training reward std, episode length.
- **Mechanism:** `gigpo_diag/*`, i.e. what each filter removed.

## Result record

Copy `results/TEMPLATE.md` for each run. Keep checkpoints and logs outside Git, and put their paths in the record. Record failed runs too.
