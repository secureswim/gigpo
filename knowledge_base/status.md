# Status and decision ledger

Last update: 16 November 2026 (merge of the Omni smoke work and the extension work into `main`).

## Where things stand

- **Next action:** `RUNBOOK_OMNI.md` step 1 on Omni.
- **Results:** no training result exists yet. The only GPU evidence is the reduced-settings smoke run below.

## Done

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
- **Tests:** 37 CPU tests (`python3 -m pytest -q tests`). They cover:
  - Upstream equivalence, filters and diagnostics.
  - Full Hydra composition of every condition through the real launch scripts.
  - The protected-settings guard and the data-preparation skip.
  - The analysis tools, and the patched trainer's advantage path with a real `DataProto`.

## Decisions

- **Platform:** Omni (Slurm). One full GPU (≥ 80 GB, not MIG) per job. Resources are fixed in `cluster/omni/resources.env` (runbook step 3, approval H1).
- **Experiment settings:** exactly the pinned upstream ALFWorld scripts: 16 tasks × 8 rollouts, minibatch 256, lr 1e-6, γ = 0.95 and ω = 1 with `mean_std_norm` for GiGPO, 150 steps, validation every 5 steps at temperature 0.4. Only hardware knobs (GPU count, micro-batch, vLLM memory share, validation batch, reference offload) may differ from the paper, identically for all runs.
- **One code base:** all conditions, including the GiGPO control and GRPO, run from the patched copy `work/verl-agent-ext`. The pristine snapshot is the reference.
- **Checkpoints:** every 5 steps, keep 2, automatic resume, and runs longer than the time limit are chained.
- **Efficiency threshold:** 0.70 validation success, fixed before any run.
- **Extension thresholds:** a fixed rule applied to the control run's training batches, steps 1–20, with a null-result gate at 5% (`knowledge_base/extension_design.md`).
- **Budget:** decided at approval H2 from the measured smoke step time (`results/decisions.md`).

## Known uncertainties

- **Omni details:** the full-GPU partitions, their limits, the credit price per GPU-hour and the balance command are discovered in runbook step 3.
- **Memory:** host-RAM needs of the paper settings are unmeasured. Each run keeps 256 ALFWorld processes alive. The runbook's "Fitting a run" ladder handles shortfalls.
- **Validation noise:** sampling at temperature 0.4 draws new tasks each round, so single points are noisy. Use the mean of the last 3 points and paired seeds.
- **Resume:** resuming restarts the environment random state, so note resumes in run records.
- **Upstream behaviour:** `adjust_batch` randomly drops a few samples for batch divisibility. This affects all conditions equally.
- **Exact training commit:** the authors do not name the exact commit used for the paper. This snapshot is the last one before their update.
- **TurtleBot3:** the Waffle Pi has no arm, so a demo would be navigation-only unless the OpenMANIPULATOR is added. It is outside the runbook.
