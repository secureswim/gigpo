# Status and decision ledger

Last update: 11 November 2026.

## Completed and verified without a GPU

- Read the proposal. Deliverables: GiGPO vs GRPO reproduction on ALFWorld, a diagnostic-first anchor-group extension, and an optional TurtleBot3 demonstration.
- Pinned `langfengQ/verl-agent` at `f974dc5977908d6007ad178c9f3c48f0c7b39331` (parent of the 3 June 2025 update). **All 1,331 snapshot files verified byte-identical** to the official commit.
- **Fixed `UPSTREAM_SHA256SUMS.txt`.** The previous hashes did not match the snapshot, so `check_baseline_scripts.py` and the old smoke script failed at their first step. The file now also covers every file the extension patches.
- Confirmed from the paper (Table 1): the 86.7% target is GiGPO **w/ std**, which is the `mean_std_norm` setting in the pinned script. GRPO is 72.8%. Both are means over 3 seeds on 2×H100.
- Confirmed that Hydra applies the last value of a repeated override, so appending overrides to the upstream scripts works.
- **Fixed a launch crash.** The upstream ALFWorld scripts read the engine from `$1` but forward all of `$@` to Hydra, so `bash run_alfworld.sh vllm <overrides>` (the old smoke command) fails with `Error parsing override 'vllm'`. The patched copies forward `"${@:2}"`. `tests/test_launch_compose.py` runs every condition through the real scripts and composes the full Hydra config to catch this kind of error without a GPU.
- Built the extension (`extension/gigpo_ext.py`, `extension/verl-agent-ext.patch`): anchor-group diagnostics, threshold filter, divergence weighting, an environment-step counter and entropy logging. With `group_filter=none` it is bit-identical to upstream.
- Wrote `scripts/run_condition.sh` (all conditions, paired seeds, checkpoints every 10 steps, auto-resume, provenance), `setup/install.sh`, `setup/preflight.sh`, `setup/build_ext_tree.sh`, the smoke and resume test with `week1/check_smoke.py`, and the analysis pipeline (`analysis/`).
- CPU test suite: `python3 -m pytest -q tests` (see the test files for coverage).
- Checked the newer upstream code (`main` at `20bd331`): it contains similarity-based anchor grouping (`enable_similarity`, `similarity_thresh`), which is absent from the pinned snapshot. Dynamic sampling **is** in the pinned snapshot. Both are recorded in `extension_design.md`.

## Pending on the Linux GPU machine

1. `bash setup/install.sh`, then commit `setup/env_lock.txt`.
2. `bash setup/preflight.sh` (all PASS).
3. `bash week1/smoke_gigpo.sh` (SMOKE PASSED), and record `timing_s/step`.
4. Runs 1–2 (GiGPO, GRPO, seed 0), then the diagnostic, then runs 3–5. See `compute_and_team.md`.
5. Optional: download the authors' W&B runs with `analysis/fetch_wandb.py`. This needs W&B access from the machine, and the run paths must be copied from the report links.

## Decisions

- **Primary task:** text ALFWorld, Qwen2.5-1.5B-Instruct. WebShop, 7B and hardware are outside the minimum result.
- **One code base:** every condition, including the GiGPO control and GRPO, runs from the patched copy `work/verl-agent-ext`. The pristine snapshot stays as the reference, and the equivalence is tested.
- **Fair comparison:** identical shared overrides for all conditions (tested). Method-specific settings stay as upstream.
- **Checkpoints:** every 10 steps, keep 2, auto-resume, the same for all conditions.
- **Success threshold for efficiency:** 0.70 validation success, fixed before any run.
- **Extension thresholds:** predeclared rule from the control run's training batches (steps 1–20). Null-result gate at 5% of signal-carrying samples.
- **Divergence weight (condition c):** w = s/(s+τ)·(n−1)/n, then mass-renormalised. Defined in `extension_design.md`.
- **Similarity-based grouping:** related work only. It changes group formation; our work changes group use.

## Known uncertainties

- The authors recommend pre-update code but do not name this exact commit as their training commit.
- The install versions follow the pinned README. flash-attn and vLLM compatibility on a particular machine can only be checked there. `transformers` is not pinned upstream, so the installed version is recorded in the lock file.
- Validation samples at temperature 0.4 and draws new tasks each round, so single points are noisy. Use the mean of the last 3 points and multiple seeds.
- Resuming from a checkpoint restarts the environment RNG.
- `adjust_batch` randomly drops a few step samples to fit batch divisibility (upstream behaviour). This affects all conditions equally.
- The TurtleBot3 Waffle Pi has no arm. A literal fetch-and-place needs the OpenMANIPULATOR; navigation-only is the narrower demo.
