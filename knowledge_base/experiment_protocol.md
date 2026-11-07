# Baseline and evaluation protocol

## Gate 1: environment

Complete `week1/gpu_machine_handoff.md`. Confirm the expected source hash and checksum check. Record package versions and any changes needed to install the pinned code. Run the ALFWorld reset/step check before downloading the model for training.

## Gate 2: one-epoch smoke run

Run `bash week1/smoke_gigpo.sh`. Inspect the log for a complete rollout, a parameter update, and validation. Record wall time and failure details. Do not compare the smoke-run score to the paper result.

## Gate 3: matched baselines

Run the pinned GiGPO and GRPO scripts using the same model, ALFWorld mode and split, environment seed, group size, data sizes, maximum episode length, optimizer budget, validation settings, and hardware. Algorithm-specific advantage parameters are expected to differ. Record any resource-motivated modification identically for both methods. Preserve exact commands and logs for every run.

The supplied scripts request 150 epochs, validate every five training steps, and set `trainer.save_freq=-1`. Choose whether to add periodic checkpoints before starting the long runs. If you add them, apply the same policy to both methods. Never treat a truncated run as a direct reproduction of the published final score.

## Outcome measures

- **Primary:** ALFWorld validation success rate, using the metric actually emitted by the pinned trainer.
- **Learning curve:** success rate versus environment interactions or training steps, with a consistent x-axis across methods.
- **Efficiency:** environment steps to a fixed success threshold selected before viewing extension results. If a method never reaches it, report that rather than inventing a value.
- **Diagnostics:** episode length, entropy, reward variance, and gradient norms where the trainer logs them. Add instrumentation only if needed; document any new metric definition.

The supplied validation samples actions at temperature `0.4`. Record seeds and repeat evaluations when feasible. A single stochastic validation point is weak evidence of a small gain. Use the same evaluation protocol for all conditions, and report run-to-run variation if multiple training seeds fit the budget.

## Result record

Copy `results/TEMPLATE.md` for each run and fill it as the run finishes. Keep large checkpoints, downloaded models, and logs outside Git; the template should contain paths or links to them. Include failed runs in the record so the experimental history is understandable.
