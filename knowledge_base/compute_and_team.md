# Compute budget and work split

## Run list in priority order

| # | Command | Why | Needed for |
| --- | --- | --- | --- |
| 1 | `SEED=0 bash scripts/run_condition.sh gigpo` | reproduction, control, and source of the diagnostic | everything |
| 2 | `SEED=0 bash scripts/run_condition.sh grpo` | baseline | Phase 1 |
| 3 | `source results/diagnostic/thresholds.env && SEED=0 bash scripts/run_condition.sh gigpo_threshold` | condition (b) | Phase 2 (only if the gate passes) |
| 4 | `source results/diagnostic/thresholds.env && SEED=0 bash scripts/run_condition.sh gigpo_weighted` | condition (c) | Phase 2 |
| 5 | `SEED=1 …` for gigpo and the best extension condition | seed variation | credible comparison |
| 6 | `SEED=0 bash scripts/run_condition.sh gigpo_dynamic` | episode-level filter reference | optional |
| 7 | `SEED=1 bash scripts/run_condition.sh grpo` | baseline variation | optional |

Runs 1 and 2 can go in parallel on separate GPU pairs. The diagnostic needs only the first 20 steps of run 1, so runs 3 and 4 can start once run 1 has passed step 20.

## Estimating the budget

The paper used 2×H100 for 150 iterations at 1.5B. Do not guess the run time: take `timing_s/step` from the smoke run.

```
GPU-hours per run ≈ seconds_per_step × 150 × N_GPUS / 3600
total             ≈ GPU-hours per run × number of runs   (4 minimum, 7 recommended)
```

The smoke step is not representative: early policies take longer episodes, and validation runs every step in the smoke test. Treat the result as an order of magnitude and re-estimate after 10 steps of run 1.

## Hardware knobs (keep them identical across conditions)

- `N_GPUS` (TP defaults to min(N_GPUS, 2)). 1 GPU works with `N_GPUS=1`, at a cost in time.
- `MICRO_BS` (default 32). Lower it on out-of-memory errors. It changes gradient accumulation, not the algorithm.
- `GPU_MEM_UTIL` (default 0.6), the vLLM share of GPU memory.
- Disk: each checkpoint with optimizer state is large (roughly 20 GB at 1.5B), and `CKPT_KEEP=2` per run. Keep at least 100 GB free (preflight checks this).

## Suggested work split (three people)

| Workstream | Owner | Deliverables |
| --- | --- | --- |
| Reproduction | ___ | install, preflight, smoke, runs 1–2, run records, comparison with the paper and the authors' W&B logs (`analysis/fetch_wandb.py`) |
| Extension | ___ | diagnostic report (`analysis/choose_thresholds.py`), runs 3–5, mechanism analysis from `gigpo_diag/*` |
| Analysis and report | ___ | `analysis/summarize.py` tables, figures, report draft, TurtleBot3 stretch decision in week 5 |
