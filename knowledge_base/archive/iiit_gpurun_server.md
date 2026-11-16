> **ARCHIVED, DO NOT FOLLOW.** These notes describe the separate IIIT Delhi `gpurun` server (192.168.3.214). The project runs on **Omni**; follow [`RUNBOOK_OMNI.md`](../../RUNBOOK_OMNI.md). The smoke command quoted below (`bash week1/smoke_gigpo.sh` through `gpurun -g 2`) belongs to an older script that crashed with `Error parsing override 'vllm'`. Kept only as a record of that server's usage policy.

# IIIT Delhi GPU server guide

Source: the server's [usage policy](http://192.168.3.214/usage_policy.html)
and [user manual](http://192.168.3.214/user-manual.html), read on 12 November
2026. The pages identify server `192.168.3.214`, two NVIDIA H100 GPUs, and
policy version 2.0 (June 2026). Re-read them if the administrator changes the
rules. The documentation site does not describe the SSH username, login
method, operating system details, or installed packages.

## Mandatory job route

Every GPU job must be launched through `gpurun`. The manual's two-GPU pattern
is `gpurun -g 2 COMMAND`. The Week 1 script wraps the upstream GiGPO launch in
this form. Run `gpurun --status` to see quota and job status and
`gpurun nvidia-smi` to inspect the GPUs through the approved route. Do not
launch training, evaluation that uses the model, or GPU diagnostics outside
`gpurun`. Jobs queue when GPUs are occupied. A stuck job can be stopped with
`gpurun --kill <job_id>`.

## Quota and fair use

- Billing is wall-clock hours multiplied by GPU count. A two-GPU one-hour job
  consumes two GPU-hours.
- The policy table gives a general 48 GPU-hour weekly default. Its user-tier
  table separately lists faculty and PhD/RA at 100 hours, M.Tech projects at
  50 hours, and B.Tech/coursework as assigned. The actual account allocation
  must be read from `gpurun --status` rather than assumed from a table.
- Quota resets Monday at 00:00 with no carry-forward. Over-budget jobs can run
  on idle GPUs but may be preempted after a 60-second grace period.
- Request only needed GPUs, release them promptly, store project files in the
  account's home directory, back up important data, and use `tmux` or `screen`
  for remote sessions. Long jobs need checkpoints.
- Accounts are individual. Do not share credentials or bypass the quota system.

## Smoke-test sequence

1. Connect using the individual's approved account. Verify the actual login
   method with the account owner or administrator; it is not specified on the
   documentation site.
2. Clone this private repository into the account's home directory and follow
   `week1/gpu_machine_handoff.md` for software and ALFWorld setup.
3. Run the source checker and ALFWorld reset/step check. These checks do not
   request a GPU.
4. Run `gpurun --status` and `gpurun nvidia-smi`. Record the quota, allocation,
   and scheduler job details if shown.
5. From a `tmux` or `screen` session, run `bash week1/smoke_gigpo.sh`. The script
   requests two GPUs through `gpurun` and logs the one-epoch result under
   `runs/week1_smoke/`.
6. Verify a completed training update and validation result, then record the
   wall time and GPU-hours used. Preserve errors and the full log if it fails.

The two H100s match the upstream script's two-GPU configuration, but model
fit, package compatibility, queue time, and the account quota remain untested
until the cluster is accessed.
