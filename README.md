# GiGPO: ALFWorld reproduction and anchor-group study

RL course project *Efficient Step-Level Credit Assignment in GiGPO* (Siddharth Kumar, Aditi Kumar, Vansh Narula). The proposal is in [`GIGPO - RL.pptx`](GIGPO%20-%20RL.pptx).

## In one paragraph

GiGPO (Feng et al., NeurIPS 2025) trains LLM agents with two advantages: an episode-level one, as in GRPO, and a step-level one computed inside *anchor groups*, the steps from different rollouts of a task that saw the same observation. With std normalisation, every anchor group pushes the policy equally hard, whatever its return gap. This project asks whether suppressing or down-weighting step-level signal from weak anchor groups helps. The current GPU handoff runs exactly three seed-0 conditions on text ALFWorld with `Qwen/Qwen2.5-1.5B-Instruct`:

| Condition | What it does |
| --- | --- |
| `gigpo` | original GiGPO (control); also writes the anchor-group diagnostic |
| `gigpo_threshold` | zeroes the step advantage of groups whose returns differ by no more than one invalid-action penalty (return std < 0.075) |
| `gigpo_weighted` | scales each group's step advantage by `s/(s+τ)·(n−1)/n`, mass-preserving, with τ = median spread from control steps 1–20 |

The thresholds follow a fixed rule, v2, set before any run ([`knowledge_base/threshold_rule.md`](knowledge_base/threshold_rule.md)). Both variants run even if a small threshold effect is forecast. GRPO, dynamic GiGPO and additional seeds are outside this handoff.

## How to run

**Start with [`HANDOFF_SEED0.md`](HANDOFF_SEED0.md), then follow [`RUNBOOK_OMNI.md`](RUNBOOK_OMNI.md) step by step on IIIT Delhi's Omni cluster.** Agents must also read [`AGENTS.md`](AGENTS.md). Background, a glossary and an FAQ are in [`knowledge_base/README.md`](knowledge_base/README.md). No other procedure in this repository is current.

```text
Omni login node                           Omni GPU batch jobs (sbatch)
---------------                           ----------------------------
1  clone, check snapshot
2  prepare.sh (env, model, data, tests)
3  choose resources ── H1 ─────────────▶  4  preflight
                                          5  smoke (2 steps) → hours per run
6  budget ── H2 ───────────────────────▶  7  gigpo control (150 steps, chained)
8  choose_thresholds.py (after step 20)
   checks: rule v2, no "Other" band ───▶  9  gigpo_threshold + gigpo_weighted
10 summarize.py, plot_curves.py, run records, report
   H3 whenever the runbook does not decide
```

## Status (18 November 2026)

**Done:**
- The software stack ran on Omni (smoke job 10333, reduced settings, pre-merge code).
- The extension, diagnostics, checkpoint/resume, launcher, analysis tools and Omni scripts are built.
- Threshold rule v2 was adopted before any run: the old percentile rule was retired, and the diagnostic now reports return bands ([`results/decisions.md`](results/decisions.md)).
- 35 CPU tests: 32 pass locally and 3 skip without Hydra, OmegaConf or Ray; all run on Omni.

**Next:** merge branch `threshold-rule-v2` into `main`, then runbook step 1 on Omni. No training result exists yet. The full ledger is in [`knowledge_base/status.md`](knowledge_base/status.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `HANDOFF_SEED0.md` | scope, exact command sequence and pass conditions for the next agent |
| `RUNBOOK_OMNI.md` | the procedure: setup, resource choice, smoke, budget, runs, diagnostic, analysis, approval gates |
| `AGENTS.md` | rules for an AI agent operating the runbook |
| `knowledge_base/` | index and glossary (`README.md`), operator FAQ, status ledger, protocol, method and code map, extension design, reward and return structure, threshold rule, DAPO and related work, compute plan; `archive/` holds superseded notes |
| `cluster/omni/` | Omni scripts: `env.sh`, `prepare.sh`, `constraints.txt`, `resources.env`, `job.sbatch`, `submit_chain.sh` |
| `scripts/run_condition.sh` | launches one condition with matched settings; refuses changes to protected experiment settings, other conditions and nonzero seeds |
| `extension/` | `gigpo_ext.py` (diagnostics, threshold filter, divergence weight) and `verl-agent-ext.patch` (trainer hooks and fixes) |
| `setup/` | `build_ext_tree.sh` (builds the patched copy in `work/`) and `preflight.sh` (login and GPU checks) |
| `upstream-verl-agent/` | official code at commit `f974dc5` (paper era), verified byte-identical; never edited |
| `week1/` | baseline-settings checker, ALFWorld check, smoke test and its checker |
| `analysis/` | progress table, threshold rule and diagnostic (`choose_thresholds.py`), outcome table, learning curves, W&B download |
| `tests/` | CPU tests (`python3 -m pytest -q tests`) |
| `results/` | run records, decisions log, diagnostic, summary and figures; `omni-short-10333/` is the first smoke run |
| `UPSTREAM_SHA256SUMS.txt` | checksums of key upstream files, including every file the patch touches |

## Reference results

The paper (Table 1, Qwen2.5-1.5B, mean over 3 seeds, 2×H100) reports:
- **GiGPO (w/ std, our setting): 86.7 ± 1.7%** ALFWorld success.
- **GRPO: 72.8 ± 3.6%.**

These are targets to compare against, not guaranteed outcomes.

The snapshot is the parent of upstream's 3 June 2025 update, which the authors say changed results. Upstream code keeps its Apache-2.0 license and notices.

The TurtleBot3 demonstration is an optional stretch goal and is not part of the runbook.
