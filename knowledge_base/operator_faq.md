# Operator FAQ

Short answers for the agent or person who runs the GPU phase. Each answer points to the authoritative file. If an answer here ever disagrees with `RUNBOOK_OMNI.md`, the runbook wins; report the mismatch.

## Scope and order

**What exactly do I run?** Three full runs, all `SEED=0`, in this order: `gigpo` (the control), then `gigpo_threshold` and `gigpo_weighted` after thresholds are derived. Before them, two setup jobs: `preflight` and `smoke`. Nothing else: no GRPO, no `gigpo_dynamic`, no seed 1. The launchers refuse them. → `HANDOFF_SEED0.md`.

**Where are the stops?** H1 after choosing resources (step 3), H2 after the smoke budget (step 6), and H3 for anything the runbook does not decide. Wait for written approval from a team member and record it in `results/decisions.md`. → `RUNBOOK_OMNI.md`, Rules.

**Can I start the extensions before the control finishes?** Yes, once the control has passed step 20 and `analysis/choose_thresholds.py` has written `results/diagnostic/thresholds.env`. Let the control keep training. Queue the two extensions together only if H2 approved the credits.

## The thresholds

**Which values will `thresholds.env` contain?** Always `MIN_GROUP_SIZE=2` and `MIN_RETURN_STD=0.075` (fixed by rule v2), plus a data-derived `WEIGHT_TAU`, the median return std of spread groups in control steps 1–20. The first line says `rule v2`. If it does not, the wrong script version ran: stop and report. → `knowledge_base/threshold_rule.md`.

**The report says "Low threshold impact expected". Do I skip `gigpo_threshold`?** No. Run both extensions whenever thresholds exist. The forecast goes in the report only. → `AGENTS.md`, `HANDOFF_SEED0.md`.

**The band table has a nonzero "Other" row. Is that a problem?** It should be empty: with the pinned reward, a never-winning group can only differ by the 0.1 penalty, which is below 0.075. A nonzero row means the reward or penalty differs from what the code assumes. Do not change anything. Stop before step 9, include the table in your report, and ask at H3.

**The report says "only the std split is shown".** The control's `step_groups.jsonl` has no `mins`/`maxs`, so it was produced by an extension built before rule v2. The thresholds are still valid; only the bands are missing. Note it in the run record and check `EXT_BUILD_INFO` in `runs/gigpo_seed0/launches.txt` against the current `extension/gigpo_ext.py`.

**No spread groups in steps 1–20?** Finish the control, then rerun with `--steps <approved-final-step>`. Still none: write no thresholds, record the mechanism result, and go to H3. → runbook step 8.

**Can I tweak a threshold that looks odd?** Never. The rule is part of the experiment. Report the observation instead.

## Resources and failures

**The job ran out of memory.** Follow the "Fitting a run" ladder: one knob at a time, and rerun the smoke test after each change. The final knob values must be identical for all three runs. → runbook, Fitting a run.

**A job hit its time limit.** That is normal. The next job in the chain resumes from the latest checkpoint (every 5 steps). Note the resume in the run record.

**A failure is not in the failure table.** Stop and send the runbook report template. Do not patch code.

## Records and git

**What do I commit?** Only `results/`, `cluster/omni/resources.env` and `cluster/omni/env_lock.txt`, on branch `omni-runs`, never `main`. Never commit `runs/`, `cache/`, `work/` or `slurm-logs/`. → `AGENTS.md` rule 6.

**What goes in a run record?** Copy `results/TEMPLATE.md` to `results/<condition>_seed0.md`, one per run, including failed runs. Fill it from logs; do not estimate unmeasured numbers.

## Testing locally (any machine, CPU only)

`python3 -m pytest -q tests` needs numpy, torch, pandas, matplotlib and pytest. Hydra, OmegaConf and Ray are optional; without them 3 tests are skipped. On Windows, the bash-based tests call `python3`. If only `python` exists, put a small `python3` wrapper on `PATH` (for example a script containing `exec python "$@"`), or run the suite on Omni's login node, where `prepare.sh` runs it anyway.
