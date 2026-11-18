# GiGPO: ALFWorld reproduction and anchor-group study

RL course project *Efficient Step-Level Credit Assignment in GiGPO* (Siddharth Kumar, Aditi Kumar, Vansh Narula). The proposal is in [`GIGPO - RL.pptx`](GIGPO%20-%20RL.pptx).

The current GPU handoff runs exactly three seed-0 conditions on text ALFWorld with `Qwen/Qwen2.5-1.5B-Instruct`: original GiGPO, threshold-filtered GiGPO, and divergence-weighted GiGPO. The extension runs happen only if the first 20 control steps show that the filters would affect enough anchor-group samples. GRPO and additional seeds are outside this handoff.

## How to run

**Start with [`HANDOFF_SEED0.md`](HANDOFF_SEED0.md), then follow [`RUNBOOK_OMNI.md`](RUNBOOK_OMNI.md) step by step on IIIT Delhi's Omni cluster.** Agents should also read [`AGENTS.md`](AGENTS.md). No other procedure in this repository is current.

## Status (16 November 2026)

**Done:**
- The software stack ran on Omni (smoke job 10333, reduced settings, pre-merge code).
- The extension, diagnostics, checkpoint/resume, launcher, analysis tools and Omni scripts are built.
- 37 CPU tests pass.

**Next:** runbook step 1 on Omni. No training result exists yet. The full ledger is in [`knowledge_base/status.md`](knowledge_base/status.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `HANDOFF_SEED0.md` | scope, exact command sequence and pass conditions for the next agent |
| `RUNBOOK_OMNI.md` | the procedure: setup, resource choice, smoke, budget, runs, diagnostic, analysis, approval gates |
| `AGENTS.md` | rules for an AI agent operating the runbook |
| `cluster/omni/` | Omni scripts: `env.sh`, `prepare.sh`, `constraints.txt`, `resources.env`, `job.sbatch`, `submit_chain.sh` |
| `scripts/run_condition.sh` | launches one condition with matched settings; refuses changes to protected experiment settings |
| `extension/` | `gigpo_ext.py` (diagnostics, threshold filter, divergence weight) and `verl-agent-ext.patch` (trainer hooks and fixes) |
| `setup/` | `build_ext_tree.sh` (builds the patched copy in `work/`) and `preflight.sh` (login and GPU checks) |
| `upstream-verl-agent/` | official code at commit `f974dc5` (paper era), verified byte-identical; never edited |
| `week1/` | baseline-settings checker, ALFWorld check, smoke test and its checker |
| `analysis/` | progress table, outcome table, learning curves, threshold rule, W&B download |
| `tests/` | CPU tests (`python3 -m pytest -q tests`) |
| `knowledge_base/` | method and code map, protocol, extension design, compute plan, status; `archive/` holds superseded notes |
| `results/` | run records, decisions log, diagnostic, summary and figures; `omni-short-10333/` is the first smoke run |
| `UPSTREAM_SHA256SUMS.txt` | checksums of key upstream files, including every file the patch touches |

## Reference results

The paper (Table 1, Qwen2.5-1.5B, mean over 3 seeds, 2×H100) reports:
- **GiGPO (w/ std, our setting): 86.7 ± 1.7%** ALFWorld success.
- **GRPO: 72.8 ± 3.6%.**

These are targets to compare against, not guaranteed outcomes.

The snapshot is the parent of upstream's 3 June 2025 update, which the authors say changed results. Upstream code keeps its Apache-2.0 license and notices.

The TurtleBot3 demonstration is an optional stretch goal and is not part of the runbook.
