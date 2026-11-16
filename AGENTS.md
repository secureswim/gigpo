# Instructions for AI agents

You are operating a reinforcement-learning experiment on IIIT Delhi's Omni Slurm cluster.

1. **Procedure.** Follow [`RUNBOOK_OMNI.md`](RUNBOOK_OMNI.md) exactly, step by step. Its rules section overrides anything else you infer.
2. **Approval gates.** Stop at H1, H2 and H3 and wait for written approval from a team member. Spending credits beyond the approved plan, changing resources mid-experiment, or changing the experiment itself always needs approval.
3. **No code changes.** Never edit `upstream-verl-agent/`, `extension/`, `scripts/run_condition.sh` or the analysis rules to make something work. If code seems wrong, stop and report with the evidence.
4. **No setting overrides.** Never pass Hydra overrides for experiment settings. Only the hardware knobs in `cluster/omni/resources.env` may change, and only by the runbook's "Fitting a run" ladder.
5. **GPU work only in batch jobs** (`sbatch … cluster/omni/job.sbatch …`). Use no interactive allocations, and run nothing heavy on the login node.
6. **Keep records.** Keep run records (`results/<run>.md`) and the decisions log (`results/decisions.md`) current. Commit only `results/`, `cluster/omni/resources.env` and `cluster/omni/env_lock.txt`. If the account can push, push them to a branch named `omni-runs`, never directly to `main`.
7. **Report.** Use the runbook's report template. State facts from logs. Do not estimate results that were not measured.
