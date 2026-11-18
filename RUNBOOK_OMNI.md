# Omni runbook: three seed-0 GiGPO experiments

**Scope:** Run only `gigpo`, `gigpo_threshold` and `gigpo_weighted`, all with `SEED=0`. Do not submit GRPO, dynamic GiGPO or another seed. The two extension runs are conditional on the anchor-group diagnostic passing. The `preflight` and `smoke` jobs are setup checks, not experimental conditions.

This is the only procedure for running the experiments. It is written for a person or an AI agent working on IIIT Delhi's **Omni** Slurm cluster. Follow the steps in order. Do not use any other cluster, script or instruction, including the archived IIIT `gpurun` server notes.

## Rules

1. **Order.** Do the steps in order. A step is done only when its **Pass** line is true.
2. **Unexpected failures.** If something fails and the step's **If it fails** list does not cover it, stop and send a report (template at the end). Do not improvise fixes.
3. **Human gates.** **H1**, **H2** and **H3** need written approval from a team member before you continue. Record each approval in `results/decisions.md` (who, when, what).
4. **Where things run.** GPU work only runs inside a Slurm batch job started from `cluster/omni/job.sbatch`. On the login node, run only `prepare.sh`, `sinfo`/`squeue`/`sacct`, analysis scripts and git. Never run training on the login node. Never hold an interactive allocation (`salloc`, `srun --pty`) open.
5. **Do not change** `upstream-verl-agent/` or `extension/`, and do not override experiment settings. `scripts/run_condition.sh` refuses protected settings. Only the hardware knobs in [Fitting a run](#fitting-a-run) may change.
6. **Same settings everywhere.** Every condition and seed uses the same `cluster/omni/resources.env`. If it has to change after step 7 has started, stop and ask (H3).
7. **Never delete** anything under `runs/` except `runs/gigpo_seed0_smoke/`, and only when a step tells you to.
8. **After every finished or failed job,** update the run record (`results/<run>.md`, copied from `results/TEMPLATE.md`).

## What is known about Omni

This comes from the passing smoke run, Omni job 10333 on 12 November 2026 (see `results/omni-short-10333/README.md`).

- **Network:** the login node has internet. Compute nodes do not. Everything the jobs need is cached by step 2.
- **Shared home:** `$HOME` is shared between the login and compute nodes.
- **Software:** Miniconda is at `~/miniconda3`, and the module `cuda12.8/toolkit/12.8.1` exists.
- **Partition `short`:** dgxh200 nodes, where a job gets one H200 MIG slice (`3g.71gb`, about 70 GB), 10 CPUs and 64 GiB host RAM. **That is too small for the paper settings**: the full 128-trajectory profile ran out of host RAM there.
- **Billing:** jobs are billed in credits. Job 10333 cost 0.10 credits, and the group balance afterwards was 34.90.
- **Not known yet** (found in step 3): the other partitions and GPU types, their limits, the exact `--gres` syntax for a full GPU, the price per GPU-hour, and the command that shows the credit balance.

**Why host RAM matters:** a run keeps 128 training environments (16 tasks × 8 attempts) and 128 validation environments alive at the same time. Each is a separate ALFWorld process.

## Step 1. Clone and verify (login node)

```bash
git clone https://github.com/secureswim/gigpo.git
cd gigpo
python3 week1/check_baseline_scripts.py
```

**Pass:** the output shows `Pinned upstream commit: f974dc59…` and `Shared ALFWorld settings match (20 checked).`
**If it fails:** the clone is incomplete or was edited. Clone again.

All later commands run from this repository root.

## Step 2. Prepare the software (login node, about 1 hour)

```bash
mkdir -p runs && bash cluster/omni/prepare.sh 2>&1 | tee runs/prepare.log
```

This does everything that needs internet:
- Creates the conda environment at `cache/env`.
- Installs the pinned packages from `cluster/omni/constraints.txt`, including xgrammar 0.1.16 built from source and the prebuilt FlashAttention wheel.
- Builds and installs the patched tree `work/verl-agent-ext`.
- Downloads the model (revision `989aa79…`) and the ALFWorld data.
- Writes the placeholder parquet files.
- Runs `setup/preflight.sh --login`, which includes the CPU test suite, and writes `cluster/omni/env_lock.txt`.

**Pass:** the last line is `PREPARE OK`, and `runs/preflight_login.txt` has only PASS lines.

**If it fails:**
- *Miniconda not found:* install Miniconda into `~/miniconda3` (user space), or set `CONDA_BASE`. Then rerun.
- *xgrammar or vLLM build/install error:* rerun once, because network hiccups are common. If it fails again, stop and report.
- *A pip conflict involving NumPy 2:* something bypassed the constraints file. Stop and report the full pip output.
- *A preflight FAIL:* fix what that line names (it says which check failed), then rerun `prepare.sh`.

Commit `cluster/omni/env_lock.txt`.

## Step 3. Choose GPU resources (login node), then gate H1

```bash
sinfo -o "%P %a %l %D %G %m %c"
sinfo -N -o "%N %P %G %m %c %f"
scontrol show partition
```

Pick the partition and flags by these rules, in priority order:

1. **One full GPU, not a MIG slice, with at least 80 GB of GPU memory** (H100/H200/B200 class). MIG gres names contain a size such as `3g.71gb`.
2. **Host RAM:** as much as the partition allows per job, aiming for at least 256G (`--mem`).
3. **CPUs:** at least 32 (`--cpus-per-task`) if allowed, otherwise the maximum allowed.
4. **Time:** the partition's maximum (`--time`).
5. If several partitions satisfy 1–4, choose the one with the lowest credit price per GPU-hour.

Write the flags into `cluster/omni/resources.env` as `RESOURCE_FLAGS="…"`, along with `TIME_LIMIT_HOURS`. Find the credit price per GPU-hour and the balance command from Omni's documentation or `sacctmgr`/`sshare` output.

**H1 (approval needed):** report the chosen flags, the price per GPU-hour and the current balance. Wait for approval before submitting any job.

## Step 4. GPU preflight (one short job)

```bash
source cluster/omni/resources.env
sbatch $RESOURCE_FLAGS --time=00:30:00 cluster/omni/job.sbatch preflight
squeue -u $USER                       # wait until it leaves the queue
tail -n 30 slurm-logs/gigpo-<jobid>.out
```

**Pass:** the job log ends with `PREFLIGHT PASSED (--gpu)`.

**If it fails:**
- *GPU below 80 GB:* you got a MIG slice. Return to step 3.
- *ALFWorld or model check fails:* the cache is incomplete. Rerun step 2.
- *Anything else:* stop and report.

## Step 5. Smoke test with the paper settings (one job, about 1–3 hours)

```bash
source cluster/omni/resources.env
sbatch $RESOURCE_FLAGS cluster/omni/job.sbatch smoke
```

This runs one GiGPO training step, then resumes for a second step. Validation and a checkpoint happen after each step. `week1/check_smoke.py` then verifies everything.

**Pass:** the job log contains `SMOKE PASSED` and an `ESTIMATED_HOURS_PER_RUN=` line.

**If it fails with a memory error:** apply [Fitting a run](#fitting-a-run). Then delete `runs/gigpo_seed0_smoke/`, save the new knob values in `resources.env`, and submit the smoke job again.

**If it fails any other way:** use the failure table below. If nothing there matches, stop and report.

Commit `cluster/omni/resources.env` once the smoke test passes.

## Step 6. Budget for at most three runs, then gate H2

From the smoke log, take `ESTIMATED_HOURS_PER_RUN` (H) and work out:

- **credits per run** = H × price per GPU-hour
- **jobs per run** = ceil(H / TIME_LIMIT_HOURS) + 1. The extra job covers resume overhead.
- **Maximum plan:** one seed-0 GiGPO control plus one seed-0 threshold and one seed-0 weighted run, each at 150 steps. If the step-8 diagnostic gate fails, stop after the GiGPO control and report that null result.

**H2 (approval needed):** report H, credits per run, the maximum cost of three runs, and the current balance. Obtain approval for this three-run maximum before submitting the first full run. A shorter shared training budget may be approved, but it must be identical for all three compared conditions. Record the approval in `results/decisions.md`. The scope of this handoff does not include GRPO, dynamic GiGPO or seed 1.

## Step 7. GiGPO seed-0 control

```bash
source cluster/omni/resources.env
SEED=0 bash cluster/omni/submit_chain.sh gigpo <jobs per run> $RESOURCE_FLAGS
```

Do not submit another chain for this same condition while one is queued or running. The control writes `runs/gigpo_seed0/gigpo_diag/step_groups.jsonl`, which step 8 needs.

**Monitor** at least daily:

```bash
python3 analysis/progress.py
squeue -u $USER
sacct -u $USER --starttime today -o JobID,JobName,State,Elapsed,ExitCode
```

**Pass:** `progress.py` shows `gigpo_seed0` at step 150 (or the approved shared `TOTAL_EPOCHS`). Step 8 may be performed once the control has passed step 20; the control can continue training.

**If a job fails:**
- *State TIMEOUT:* normal. The next job in the chain resumes from the last checkpoint (saved every 5 steps).
- *Chain used up before the last step:* submit one more job with the same condition, seed, resources and budget.
- *Error signature in `progress.py`:* look it up in the failure table. If the same error appears twice in a row, stop and report.

## Step 8. Anchor-group diagnostic (after `gigpo_seed0` passes step 20)

```bash
source cluster/omni/env.sh
python3 analysis/choose_thresholds.py --diag runs/gigpo_seed0/gigpo_diag/step_groups.jsonl
```

This writes `results/diagnostic/{thresholds.env,diagnostic_report.md,anchor_groups.png}`. The fixed rule uses training batches only. Do not edit its output.

- **Report says "Gate passed":** finish the control and go to step 9.
- **Report says "Gate not passed":** do not run either extension condition. Report the diagnostic as a null result, then go to step 10 for analysis of the control. The launch scripts also reject extension submissions when the gate has not passed.

Commit `results/diagnostic/` to the `omni-runs` branch.

## Step 9. Two extension runs (only after the gate passed)

```bash
source cluster/omni/resources.env
SEED=0 bash cluster/omni/submit_chain.sh gigpo_threshold <jobs per run> $RESOURCE_FLAGS
SEED=0 bash cluster/omni/submit_chain.sh gigpo_weighted  <jobs per run> $RESOURCE_FLAGS
```

`job.sbatch` loads `results/diagnostic/thresholds.env` automatically. Both runs use exactly the same model, resource knobs, training budget and validation settings as the control. They may be queued at the same time if the approved credit budget and cluster policy allow it. Monitor and handle failures as in step 7.

**Pass:** `progress.py` shows `gigpo_threshold_seed0` and `gigpo_weighted_seed0` at the approved final step.

## Step 10. Analyze and hand off results

On the login node, after all authorized runs have finished:

```bash
source cluster/omni/env.sh
python3 analysis/summarize.py --runs runs --threshold 0.70 --out results/summary.csv
python3 analysis/plot_curves.py --runs runs --out results/figures --threshold 0.70
```

Fill one run record for each run that actually ran; include failures and resumes. Commit `results/`. Never commit `runs/`, `cache/`, `work/` or `slurm-logs/`. Report the final validation success and mean of the last three validation points, plus environment steps. With one seed, treat performance differences as preliminary.

**H3** applies any time a decision is needed that this runbook does not make.

## Fitting a run

These knobs change memory use and speed only, never the algorithm. Change one at a time, in the order given, and rerun the smoke test (step 5) after each change. Record the final values in `cluster/omni/resources.env`. The same values must be used for every run.

| Symptom (in `slurm-logs/*.out` or `runs/*/train.log`) | Knob, in this order |
| --- | --- |
| `CUDA out of memory`, `torch.OutOfMemoryError` | `MICRO_BS` 16 → 8 → 4 → 2 → 1, then `GPU_MEM_UTIL` 0.5 → 0.4 |
| `No available memory for the cache blocks` (vLLM) | `GPU_MEM_UTIL` 0.5 → 0.6 → 0.7, only if no CUDA OOM occurred |
| Slurm `oom-kill` / `Out Of Memory`, Ray `OutOfMemoryError`, workers killed | first raise `--mem` to the partition maximum, then `VAL_BATCH` 128 → 64 → 32 → 16 |
| Host RAM still exhausted at `VAL_BATCH=16` | stop and report (H3) |

`VAL_BATCH` only changes how many of the 128 validation tasks run at once; all 128 are still evaluated. `N_GPUS=2` with two GPUs per job is allowed only with H3 approval, because it doubles the cost.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| `Error parsing override 'vllm'` | the job used an unpatched tree | rerun step 2 |
| `Patched tree not found` | `work/verl-agent-ext` missing | rerun step 2 |
| `Refusing override …: protected` | a protected experiment setting was passed | remove it; settings come from `resources.env` and environment variables only |
| `huggingface.co` connection errors, `OfflineModeIsEnabled`, missing `config.json` | model not cached | rerun step 2 on the login node |
| ALFWorld `FileNotFoundError` for `json_2.1.1` | ALFWorld data missing | rerun step 2 |
| `Upstream snapshot differs` | repository files were modified | discard local changes (`git status`), or clone again |
| `results/diagnostic/thresholds.env missing` | step 8 not done | do step 8 first |
| `already reached step 150; nothing to do` | run finished | nothing; cancel the rest of its chain (`scancel`) |
| `DUE TO TIME LIMIT` | partition time limit reached | normal; the chain resumes it |
| Ray fails to start, or socket path too long | stale Ray temporary files | resubmit; `RAY_TMPDIR` is per job and is removed at exit |
| Anything else | | stop and report |

## Report template

```
Step: <number and name>          Status: PASS / FAIL / WAITING FOR H1|H2|H3
Job IDs: <ids>                    Credits used: <amount>, balance: <amount>
What ran: <exact command>
What happened: <last 20 relevant log lines or progress.py output>
Hardware knobs in effect: <resources.env values>
Decision needed (if any): <question with the options>
```
