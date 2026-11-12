# GiGPO: ALFWorld reproduction and anchor-group study

RL course project: *Efficient Step-Level Credit Assignment in GiGPO* (Siddharth Kumar, Aditi Kumar, Vansh Narula). The project reproduces **GiGPO versus GRPO** on text ALFWorld with `Qwen/Qwen2.5-1.5B-Instruct`, then tests whether selecting informative step-level anchor groups improves GiGPO. The proposal is [`GIGPO - RL.pptx`](GIGPO%20-%20RL.pptx). No training result is claimed yet.

## Status

Everything that does not need a GPU is done and tested on CPU: the pinned snapshot is verified, and the extension (diagnostics, threshold filter, divergence weighting), run scripts, install and preflight scripts, smoke and resume check, and analysis pipeline are all in place. What remains is to run on the Linux GPU machine. See the [status ledger](knowledge_base/status.md).

## On the GPU machine

```bash
git clone https://github.com/secureswim/gigpo.git && cd gigpo
bash setup/install.sh          # env, patched tree, ALFWorld data, model, lock file
bash setup/preflight.sh        # every line must PASS
bash week1/smoke_gigpo.sh      # 2 steps + resume; must print SMOKE PASSED

SEED=0 bash scripts/run_condition.sh gigpo      # control + diagnostic
SEED=0 bash scripts/run_condition.sh grpo       # baseline

python3 analysis/choose_thresholds.py --diag runs/gigpo_seed0/gigpo_diag/step_groups.jsonl
source results/diagnostic/thresholds.env        # only if the null-result gate passed
SEED=0 bash scripts/run_condition.sh gigpo_threshold
SEED=0 bash scripts/run_condition.sh gigpo_weighted

python3 analysis/summarize.py --runs runs --threshold 0.70 --out results/summary.csv
python3 analysis/plot_curves.py --runs runs --out results/figures --threshold 0.70
```

Full instructions: [GPU-machine handoff](week1/gpu_machine_handoff.md). Run order and budget: [compute and team](knowledge_base/compute_and_team.md).

## Before the GPU (any machine with Python)

```bash
pip install torch numpy pandas matplotlib pytest omegaconf   # CPU torch is enough
python3 week1/check_baseline_scripts.py
python3 -m pytest -q tests
DRY_RUN=1 EXT_TREE=work/verl-agent-ext bash scripts/run_condition.sh gigpo   # after setup/build_ext_tree.sh
```

## Repository map

| Path | Purpose |
| --- | --- |
| `upstream-verl-agent/` | Unmodified official source at commit `f974dc5` (paper-era, before the 3 June 2025 update). Never edited. |
| `extension/` | `gigpo_ext.py` (diagnostics and group selection) and `verl-agent-ext.patch` (trainer hooks) |
| `setup/` | `install.sh`, `preflight.sh`, `build_ext_tree.sh` (builds the patched copy in `work/`) |
| `scripts/run_condition.sh` | One launcher for every condition: seed, GPUs, checkpoints, resume, provenance |
| `analysis/` | log parsing, outcome table, learning curves, threshold rule, W&B download |
| `tests/` | CPU tests: upstream equivalence, filters, diagnostics, scripts, analysis |
| `week1/` | baseline checker, ALFWorld check, smoke and resume test, handoff |
| `knowledge_base/` | method and code map, protocol, extension design, compute and team, status |
| `results/` | run-record template; diagnostic report, summary and figures land here |
| `UPSTREAM_SHA256SUMS.txt` | checksums of key upstream files, including every file the patch touches |

## Why this upstream revision

The [official repository](https://github.com/langfengQ/verl-agent) says results changed after its 3 June 2025 major update and recommends an earlier version for reproducing the paper. This snapshot is the first parent of that update. All 1,331 files were verified byte-identical to the official commit. The paper reports **86.7 ± 1.7%** ALFWorld success for GiGPO (w/ std, our setting) and **72.8 ± 3.6%** for GRPO at 1.5B, averaged over 3 seeds. These are reference results, not guaranteed outcomes. Upstream code keeps its Apache-2.0 license and notices.

The TurtleBot3 demonstration is a stretch goal outside the experiment path.
