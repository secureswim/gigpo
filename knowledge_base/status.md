# Status and decision ledger

Last local audit: 12 November 2026.

## Cluster update: 12 November 2026

The single-MIG GiGPO smoke test passed in Omni job 10333: one training iteration
with a nonzero gradient norm, 128-example validation, and exit zero. Validation
success was 6.25%; successful attempt wall time was 33 minutes. Dependencies,
model and ALFWorld data are staged in an isolated environment on WEKA. See
[`results/omni-short-10333/README.md`](../results/omni-short-10333/README.md) for exact configuration,
logs, compatibility changes, earlier failures, and timing breakdowns. The smoke
profile reduces training concurrency to fit short's 64 GiB host RAM; full matched
GiGPO/GRPO runs remain pending. No trained checkpoint was saved. Historical
pending items below describe the original handoff, not the current smoke status.

## Completed and verified locally

- Read the six-slide project proposal and identified the core deliverables: GiGPO versus GRPO reproduction on ALFWorld, diagnostic-first anchor-group extension, and optional TurtleBot3 demonstration.
- Retrieved the official `langfengQ/verl-agent` repository and selected commit `f974dc5977908d6007ad178c9f3c48f0c7b39331`, the parent of the 3 June 2025 major update.
- Imported that revision as a source snapshot without its Git history. Original `LICENSE` and `Notice.txt` are retained. Key-file checksums are in `UPSTREAM_SHA256SUMS.txt`.
- Confirmed the pinned GiGPO and GRPO ALFWorld scripts agree on 20 shared settings using `python3 week1/check_baseline_scripts.py`.
- Prepared the Linux installation instructions, an ALFWorld reset/step check, and a one-epoch GiGPO smoke script. Python syntax and the shell script syntax were checked locally.
- Separated unrelated quiz application files from this repository. No quiz files are part of the handoff.
- Read the IIIT Delhi server's policy and manual. The smoke script now requests
  two GPUs through the mandatory `gpurun` scheduler. The cluster itself has not
  yet been accessed from this workspace through that separate workflow; the
  verified Omni Slurm run is recorded above.

## Pending on the Linux GPU machine

- Install the pinned Python/CUDA packages and download ALFWorld data.
- Verify the account's connection method and actual quota with `gpurun --status`.
- Execute `python3 week1/check_alfworld.py` and capture its output.
- Execute `bash week1/smoke_gigpo.sh`; verify an optimizer update and validation result, then retain `runs/week1_smoke/metadata.txt` and `train.log`.
- Record actual runtime, package deviations, hardware, and any errors.
- Run full matched GiGPO and GRPO baselines; no result exists yet.

## Decisions already made

- **Primary task:** text ALFWorld with Qwen2.5-1.5B-Instruct. WebShop, 7B models, and hardware deployment are outside the minimum viable result.
- **Code baseline:** pre-update paper-era snapshot. Do not mix newer repository files or newer vLLM instructions into it without documenting an explicit compatibility change.
- **Fair comparison:** same model, environment, rollout count, update budget, and evaluation procedure for GiGPO and GRPO. Keep GiGPO's algorithm-specific gamma and step-advantage settings.
- **Extension gate:** measure anchor-group signal before filtering. A zero or tiny fraction of degenerate groups is a valid null diagnostic.
- **Source history:** the upstream code is a vendored snapshot so the handoff repository can have its own Git history. Provenance and upstream authorship are retained here, in the upstream license, and in the source comments.

## Known uncertainties

- The authors recommend pre-update code but do not identify this exact commit as their paper training commit. It is the last mainline revision before the update.
- The original run uses stochastic validation (`temperature=0.4`, sampling enabled); one evaluation can vary. Repeated evaluations or multiple seeds strengthen comparisons.
- The supplied scripts disable periodic checkpoint saving (`trainer.save_freq=-1`). Decide on a checkpoint policy before long runs and apply it consistently.
- The proposal's TurtleBot3 Waffle Pi pipeline needs a separate manipulation mechanism for a literal fetch-and-place demo. A navigation-only high-level policy demo is the narrower option.
