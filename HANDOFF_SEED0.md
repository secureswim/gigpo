# GPU handoff: seed-0 GiGPO study

**Read this page, then follow [RUNBOOK_OMNI.md](RUNBOOK_OMNI.md) in order.** Work on IIIT Delhi's Omni cluster. This branch is the experiment handoff; no full training result exists yet.

## Authorized experiment scope

Run exactly these three full training conditions with `SEED=0` and one shared training budget (default: 150 steps):

1. `gigpo`: original GiGPO control and anchor-group diagnostics.
2. `gigpo_threshold`: remove step advantage from small or low-spread anchor groups.
3. `gigpo_weighted`: reweight step advantage by group size and return spread.

Older notes mention `grpo`, `gigpo_dynamic` and seed 1; all are outside this handoff. Do not submit them. `preflight` and `smoke` are mandatory setup checks, not experimental conditions. The launchers reject unsupported conditions and nonzero seeds.

## Before any full run

1. Clone `main`: `git clone https://github.com/secureswim/gigpo.git` and `cd gigpo`. Check that `analysis/choose_thresholds.py` contains `RULE_VERSION = "v2"`. If it does not, the threshold-rule update has not been merged yet; stop and ask.
2. On the login node, run the preparation and checks in runbook steps 1–2. Do not run training on the login node.
3. Inspect Omni resources and fill `cluster/omni/resources.env` in step 3. **Its allocation flags are blank in the repository.** Obtain H1 approval before any GPU job.
4. Run the GPU preflight and two-step smoke test (steps 4–5). Use its measured `ESTIMATED_HOURS_PER_RUN` to calculate `<jobs-per-run>` and maximum credits for three runs. Obtain H2 approval for the budget before full training. If H2 sets a shorter `TOTAL_EPOCHS`, export that same value for all three runs and record it.

## Exact full-run order

The following commands run from the repository root. Replace `<jobs-per-run>` with the positive integer calculated in runbook step 6. `RESOURCE_FLAGS` comes from the approved `resources.env`.

```bash
source cluster/omni/resources.env
SEED=0 bash cluster/omni/submit_chain.sh gigpo <jobs-per-run> $RESOURCE_FLAGS
```

Monitor with `python3 analysis/progress.py`. When `gigpo_seed0` has reached step 20, run on the login node:

```bash
source cluster/omni/env.sh
python3 analysis/choose_thresholds.py --diag runs/gigpo_seed0/gigpo_diag/step_groups.jsonl
```

The command should create `results/diagnostic/thresholds.env` and `diagnostic_report.md`. Under threshold rule v2 ([`knowledge_base/threshold_rule.md`](knowledge_base/threshold_rule.md)), `thresholds.env` always holds `MIN_GROUP_SIZE=2` and `MIN_RETURN_STD=0.075`, plus `WEIGHT_TAU` = the median return spread of the control's spread groups in steps 1–20. Do the checks listed in runbook step 8 before continuing. **Run both variants even if the report says "Low threshold impact expected."** That is a forecast for thresholding, not a stop rule for weighting. Let the GiGPO control finish before comparing outcomes.

```bash
source cluster/omni/resources.env
SEED=0 bash cluster/omni/submit_chain.sh gigpo_threshold <jobs-per-run> $RESOURCE_FLAGS
SEED=0 bash cluster/omni/submit_chain.sh gigpo_weighted  <jobs-per-run> $RESOURCE_FLAGS
```

The two extension chains may be queued together only if H2 approved enough credits and Omni allows the allocations. Do not submit a second chain for the same condition while one is active. A time-limited job resumes from the latest checkpoint in the next chained job.

## If threshold calculation finds no spread

If the 20-step check exits without `thresholds.env`, continue `gigpo_seed0` to its approved final step. Repeat the diagnostic using `--steps <approved-final-step>`. This training-only fallback is fixed before looking at validation outcomes.

If the full control still has no groups with different returns, do not invent thresholds or submit near-identical variants. Write up the mechanism result and request H3 scope approval. A seed-0 GRPO run would provide a matched paper baseline; a second GiGPO seed would test whether the result was seed-specific. Neither is authorized by this three-run handoff.

## What to save and report

- `runs/gigpo_seed0/`, `runs/gigpo_threshold_seed0/`, `runs/gigpo_weighted_seed0/`: local logs, checkpoints and diagnostics. Keep these out of Git.
- `results/diagnostic/`: anchor-group report (including the return-band table) and the thresholds actually used.
- `results/<condition>_seed0.md`: one record per run, including failures, resumes, allocation and exact command.
- After the runs: `python3 analysis/summarize.py --runs runs --threshold 0.70 --out results/summary.csv` and `python3 analysis/plot_curves.py --runs runs --out results/figures --threshold 0.70`.

Compare held-out validation success (final and mean of the last three checks), steps and environment interactions to the 70% reference, learning curves and stability metrics. The variants use the same rollouts per step, so the objective is better learning, not shorter time for a fixed 150-step job. With one seed, describe any gain as preliminary.
