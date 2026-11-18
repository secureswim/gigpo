# Status and decision ledger

Last update: 18 November 2026 (threshold rule v2 and handoff knowledge base, branch `threshold-rule-v2`; merge to `main` before the GPU handoff).

**Current plan:** run only seed-0 `gigpo`, `gigpo_threshold` and `gigpo_weighted` as described in `HANDOFF_SEED0.md` and `RUNBOOK_OMNI.md`. The 16 November setup history below remains useful background; its older plans for GRPO and extra seeds are superseded. No full training run has started.

## Where things stand

- **Next action:** merge `threshold-rule-v2` into `main`, then `RUNBOOK_OMNI.md` step 1 on Omni.
- **Results:** no training result exists yet. The only GPU evidence is the reduced-settings smoke run below.

## Done

### Threshold rule v2 (Aditi Kumar with Claude; 18 November 2026)

- **Rule v1 reviewed and retired before any run.** v1 used `MIN_GROUP_SIZE=3` and `MIN_RETURN_STD` = p25 of spread-group std. In text ALFWorld, returns take discrete values (`10*0.95^k`, 0, minus 0.1 for an invalid action), so the spread distribution is multimodal and a percentile has no fixed meaning. The size rule also removed every size-2 success-vs-failure pair. See `threshold_rule.md` and `reward_and_return_structure.md`.
- **Rule v2:** `MIN_GROUP_SIZE=2` (no size filter), `MIN_RETURN_STD=0.075` fixed (just above 0.1/sqrt(2), the largest std of a group that differs only by one invalid-action penalty), `WEIGHT_TAU` = median as before.
- **Diagnostics:** `extension/gigpo_ext.py` now also records per-group `mins`/`maxs` (diagnostic only; advantages unchanged). `analysis/choose_thresholds.py` reports return bands (mixed outcome / timing / penalty-scale), drift by 5-step block, the forecast and the retired v1 share, and plots a log-scale histogram by band.
- **Tests:** new tests pin the rule's constants to the upstream script and check bands, forecast and the v2 filter on real return values. A stale test that still launched `grpo`, `gigpo_dynamic` and seed 1, and so failed against the scope-restricted launcher, now checks the three seed-0 conditions and that out-of-scope launches are refused.
- **Knowledge base:** added `README.md` (index and glossary), `operator_faq.md`, `threshold_rule.md`, `reward_and_return_structure.md` and `dapo_and_related_work.md`.

### On Omni (Siddharth Kumar, Vansh Narula; 12 November 2026)

- **Smoke test passed** (job 10333; see `results/omni-short-10333/README.md`):
  - Hardware: one H200 MIG slice (`3g.71gb`) on partition `short`, 64 GiB host RAM.
  - Ran one GiGPO training step: gradient norm 4.306, exit 0, 6.25% validation success, 30.6 min.
  - Settings were reduced: 4 tasks per step instead of 16, PPO minibatch 32 instead of 256, validation batches of 8. It used the pristine snapshot, without the extension.
  - A full 128-trajectory attempt ran out of the 64 GiB host RAM.
- **Package fixes that make vLLM 0.8.2 install** (now in `cluster/omni/prepare.sh` and `constraints.txt`):
  - xgrammar 0.1.16 built from its official source tag.
  - NumPy 1.26.4 with thinc 8.2.5, spaCy 3.7.5 and matching pyarrow, pandas, SciPy and plotting versions.
  - The prebuilt FlashAttention 2.7.4.post1 wheel.
- **The IIIT `gpurun` server** (2×H100) policy was documented. That server is not used; the notes are archived in `knowledge_base/archive/iiit_gpurun_server.md`.

### Code and tests (Aditi Kumar; 11–16 November 2026)

- **Snapshot verified:** all 1,331 files of `upstream-verl-agent/` match official commit `f974dc5` byte for byte.
- **Fixed: checksum mismatch.** `UPSTREAM_SHA256SUMS.txt` had been generated from CRLF copies, so `check_baseline_scripts.py` failed on Linux. The manifest now holds the LF hashes and covers every file the extension patches. The checker also accepts the old CRLF form (fix from the Omni side).
- **Fixed: launch crash.** The upstream ALFWorld scripts forward the engine name `vllm` to Hydra, causing `Error parsing override 'vllm'`. The patched copies forward `"${@:2}"`.
- **Extension built** (`extension/gigpo_ext.py` and `extension/verl-agent-ext.patch`):
  - Anchor-group diagnostics.
  - The threshold filter and the divergence weight. With `group_filter=none` the advantages are bit-identical to upstream.
  - Logging of `actor/entropy` and `env/steps_collected`.
  - The data-preparation step is skipped on compute nodes, which are offline, when its files already exist.
- **Omni workflow:** `cluster/omni/` (environment, one install script, Slurm job, job chains), `RUNBOOK_OMNI.md` with approval gates, `AGENTS.md`, and the progress, analysis and threshold tools.
- **Launcher safeguards:** `scripts/run_condition.sh` refuses overrides of protected experiment settings. Only hardware knobs can change.
- **Tests:** 35 CPU tests (`python3 -m pytest -q tests`; 32 pass and 3 skip when Hydra, OmegaConf or Ray are missing, all run on Omni). They cover:
  - Upstream equivalence, filters and diagnostics.
  - Full Hydra composition of every condition through the real launch scripts.
  - The protected-settings guard and the data-preparation skip.
  - The analysis tools, and the patched trainer's advantage path with a real `DataProto`.

## Decisions

- **Platform:** Omni (Slurm). One full GPU (≥ 80 GB, not MIG) per job. Resources are fixed in `cluster/omni/resources.env` (runbook step 3, approval H1).
- **Experiment settings:** exactly the pinned upstream ALFWorld scripts: 16 tasks × 8 rollouts, minibatch 256, lr 1e-6, γ = 0.95 and ω = 1 with `mean_std_norm` for GiGPO, 150 steps, validation every 5 steps at temperature 0.4. Only hardware knobs (GPU count, micro-batch, vLLM memory share, validation batch, reference offload) may differ from the paper, identically for all runs.
- **One code base:** all three active GiGPO-family conditions run from the patched copy `work/verl-agent-ext`. The pristine snapshot is the reference.
- **Checkpoints:** every 5 steps, keep 2, automatic resume, and runs longer than the time limit are chained.
- **Efficiency threshold:** 0.70 validation success, fixed before any run.
- **Extension thresholds (rule v2, 18 Nov 2026):** `MIN_GROUP_SIZE=2`, `MIN_RETURN_STD=0.075` fixed from the reward structure, `WEIGHT_TAU` = median spread-group std of the control's training steps 1–20. A <5% threshold-impact forecast is reported but does not stop either extension. If no return spread appears, retry with the full approved control budget (`knowledge_base/threshold_rule.md`).
- **Budget:** decided at approval H2 from the measured smoke step time (`results/decisions.md`).

## Known uncertainties

- **Omni details:** the full-GPU partitions, their limits, the credit price per GPU-hour and the balance command are discovered in runbook step 3.
- **Memory:** host-RAM needs of the paper settings are unmeasured. Each run keeps 256 ALFWorld processes alive. The runbook's "Fitting a run" ladder handles shortfalls.
- **Validation noise:** sampling at temperature 0.4 draws new tasks each round, so single points are noisy. Use the mean of the last 3 points; all active conditions share seed 0, and any gain is preliminary.
- **Resume:** resuming restarts the environment random state, so note resumes in run records.
- **Upstream behaviour:** `adjust_batch` randomly drops a few samples for batch divisibility. This affects all conditions equally.
- **Weighted condition emphasis:** on synthetic data with the real return structure, tau = median gives timing-only groups a small weight. After mass rescaling, `gigpo_weighted` concentrates the micro advantage on success-vs-failure groups and weakens the "be faster" signal. Check against the real diagnostic, and keep it in mind for episode-length curves.
- **Threshold representativeness:** `WEIGHT_TAU` comes from steps 1–20, when the success rate is low. The report's block table shows drift; it is reported, not acted on.
- **Exact training commit:** the authors do not name the exact commit used for the paper. This snapshot is the last one before their update.
- **TurtleBot3:** the Waffle Pi has no arm, so a demo would be navigation-only unless the OpenMANIPULATOR is added. It is outside the runbook.
