# Compute budget and work split

All runs happen on Omni through [`RUNBOOK_OMNI.md`](../RUNBOOK_OMNI.md). Each command below is run from the repository root, after `source cluster/omni/resources.env`.

## Run list in priority order

| # | Command | Why | When |
| --- | --- | --- | --- |
| 1 | `SEED=0 bash cluster/omni/submit_chain.sh gigpo <jobs> $RESOURCE_FLAGS` | reproduction, control, and source of the diagnostic | runbook step 7 |
| 2 | `SEED=0 bash cluster/omni/submit_chain.sh grpo <jobs> $RESOURCE_FLAGS` | baseline | step 7, in parallel with run 1 |
| 3 | `SEED=0 bash cluster/omni/submit_chain.sh gigpo_threshold <jobs> $RESOURCE_FLAGS` | condition (b) | step 9, once run 1 passes step 20 and the gate passes |
| 4 | `SEED=0 bash cluster/omni/submit_chain.sh gigpo_weighted <jobs> $RESOURCE_FLAGS` | condition (c) | step 9 |
| 5 | `SEED=1 …` for `gigpo` and the better extension condition | seed variation | step 10, if approved at H2 |
| 6 | `SEED=0 … gigpo_dynamic` | episode-level filter reference | optional, if approved |
| 7 | `SEED=1 … grpo` | baseline variation | optional, if approved |

`<jobs>` is ceil(hours per run / `TIME_LIMIT_HOURS`) + 1 (runbook step 6).

## Budget

`week1/check_smoke.py` prints `ESTIMATED_HOURS_PER_RUN` from the smoke run's measured step time. It counts validation every 5 steps and excludes queue time.

```
credits per run = ESTIMATED_HOURS_PER_RUN × N_GPUS × credit price per GPU-hour
plan cost       = credits per run × number of runs
```

The team approves the plan at H2 and records it in `results/decisions.md`.

**What is known so far:**
- The only measurement is the reduced smoke run (job 10333): 30.6 min for one step with a quarter of the batch, two-thirds of it validation in batches of 8. It is not a basis for the budget.
- Job 10333 cost 0.10 credits, and the group balance afterwards was 34.90.
- The paper trained on 2×H100. A rough expectation for one full GPU is tens of hours per run. The smoke measurement replaces this.

## Hardware

- **Per job:** one full GPU (≥ 80 GB, not MIG). Host RAM as high as the partition allows: each run keeps 256 ALFWorld processes alive.
- **Knobs** live in `cluster/omni/resources.env`, change only by the runbook's "Fitting a run" ladder, and stay identical for every run.
- **Disk:** each checkpoint with optimizer state is roughly 20 GB, and 2 are kept per run. Keep at least 100 GB free (preflight checks this).

## Suggested work split

| Workstream | Owner | Deliverables |
| --- | --- | --- |
| Omni operation | ___ (or an agent under `AGENTS.md`) | runbook steps 1–10, run records, progress reports |
| Approvals and extension analysis | ___ | H1–H3 decisions, diagnostic report review, mechanism analysis from `gigpo_diag/*` |
| Analysis and report | ___ | `analysis/summarize.py` tables, figures, comparison with the authors' W&B logs (`analysis/fetch_wandb.py`, needs internet), report draft |
