# Cluster scripts

All experiments run on IIIT Delhi's **Omni** Slurm cluster. Follow [`RUNBOOK_OMNI.md`](../RUNBOOK_OMNI.md); this folder only holds the scripts it calls.

| File | Where it runs | Purpose |
| --- | --- | --- |
| `omni/env.sh` | everywhere (sourced) | paths, conda environment, model revision, offline settings inside jobs |
| `omni/prepare.sh` | login node | one-time install, downloads and checks; ends with `PREPARE OK` |
| `omni/constraints.txt` | login node | package pins that make vLLM 0.8.2 work (NumPy 1.26.4, thinc 8.2.5, …) |
| `omni/resources.env` | submit shell (sourced) | Slurm resource flags and hardware knobs shared by every job |
| `omni/job.sbatch` | compute node | one batch job: `preflight`, `smoke` or a training condition |
| `omni/submit_chain.sh` | login node | submits a run as a chain of resuming jobs |
| `omni/env_lock.txt` | — | exact package versions, written by `prepare.sh` |

## History

The first Omni smoke run (job 10333, 12 November 2026) passed using earlier scripts, `prepare-omni.sh`, `smoke-omni.sh` and `run-in-job.sh`, with reduced batch settings on a MIG slice. Its record is in [`results/omni-short-10333/`](../results/omni-short-10333/README.md). Those scripts were replaced by `omni/`:

- **Installation:** the package fixes they found are kept unchanged in `prepare.sh` and `constraints.txt`. These are xgrammar 0.1.16 built from source, NumPy 1.26.4 with thinc 8.2.5 and spaCy 3.7.5, and the prebuilt FlashAttention wheel.
- **Paths:** the old scripts hard-coded one account's home directory. The new ones work for any account.
- **Patched tree:** they installed the pristine snapshot. The new ones install the patched tree, so the extension and diagnostics are active.
- **Missing helpers:** they called helper scripts (`job-step.sh`, `inspect-gpu.sh`) that were never committed.
