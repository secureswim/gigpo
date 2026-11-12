# GiGPO: ALFWorld reproduction and anchor-group study

This repository is the handoff for a reinforcement-learning course project. The core experiment reproduces **GiGPO versus GRPO** on text ALFWorld using `Qwen/Qwen2.5-1.5B-Instruct`. The proposed extension asks whether selecting informative step-level anchor groups improves GiGPO. The proposal is included as [`GIGPO - RL.pptx`](GIGPO%20-%20RL.pptx). A single-iteration GiGPO smoke test has passed on Omni; full matched baseline results remain pending.

## Current status

**12 November 2026: the GiGPO smoke test passed on Omni job 10333.** It completed one training iteration and validation on 128 examples with exit code 0. The successful attempt took 33 minutes 10 seconds and reported 6.25% validation success. This verifies the pipeline; it is not a full-training result.

See the **[smoke-test README](results/omni-short-10333/README.md)** for batch sizes, GPU/RAM allocation, exact configuration, package versions, timing breakdown, fixes, and committed logs. Full matched GiGPO/GRPO training remains pending. See the [status ledger](knowledge_base/status.md) for the remaining work.

The separate IIIT Delhi `gpurun` workflow is documented in the [cluster guide](knowledge_base/cluster_guide.md). The verified Omni Slurm workflow and its single-MIG profile are in [cluster/README.md](cluster/README.md).

## Quick start on the Linux machine

The repository is private. Ask the owner for collaborator access, then clone it and work from its root. Do not update `upstream-verl-agent/` before the baseline runs.

```bash
git clone https://github.com/secureswim/gigpo.git
cd gigpo
```

```bash
python3 week1/check_baseline_scripts.py
```

Read the [cluster guide](knowledge_base/cluster_guide.md) and follow [the GPU-machine handoff](week1/gpu_machine_handoff.md) to install the pinned dependencies and download ALFWorld. Then run:

```bash
python3 week1/check_alfworld.py
bash week1/smoke_gigpo.sh
```

The smoke script runs the GPU command through the cluster's mandatory `gpurun -g 2` wrapper and captures package versions and console output under `runs/week1_smoke/`. Check that the log shows a completed training update and validation result. The one-epoch run only verifies the pipeline. It cannot establish the paper's success rate.

## Repository map

| Path | Purpose |
| --- | --- |
| `upstream-verl-agent/` | Unmodified source snapshot from the official repository at commit `f974dc5977908d6007ad178c9f3c48f0c7b39331` |
| `week1/` | Baseline settings, Linux handoff, static checker, ALFWorld check, and GiGPO smoke run |
| `knowledge_base/` | Method explanation, code map, cluster guide, experiment protocol, extension design, decisions, and status |
| `results/TEMPLATE.md` | Record for each GPU run; copy and fill it after execution |
| `UPSTREAM_SHA256SUMS.txt` | Checksums for key files in the upstream snapshot |

## Why this upstream revision

The [official repository](https://github.com/langfengQ/verl-agent) says results changed after its 3 June 2025 major update and recommends a version before it to reproduce the original paper. This snapshot is the first parent of that update. The paper reports **86.7% ALFWorld success** for GiGPO with Qwen2.5-1.5B-Instruct. This number is a reference result, not a guaranteed outcome. The exact training environment and every deviation must be logged. The source code remains under its original Apache-2.0 license and notices in `upstream-verl-agent/`.

## Handoff order

1. Read [status](knowledge_base/status.md), [method and code map](knowledge_base/method_and_code.md), and the [experiment protocol](knowledge_base/experiment_protocol.md).
2. Complete the [GPU-machine handoff](week1/gpu_machine_handoff.md).
3. Preserve the smoke-run log and fill a [run record](results/TEMPLATE.md).
4. Launch matched GiGPO and GRPO baselines only after the smoke check passes.
5. Inspect anchor groups before implementing or training the extension. Use [extension design](knowledge_base/extension_design.md).

The TurtleBot3 demonstration in the proposal is a stretch goal. It is outside the baseline and extension experiment path.
