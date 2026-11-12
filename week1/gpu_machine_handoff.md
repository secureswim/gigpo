# Linux GPU-machine handoff

Everything below runs from the project root on the Linux machine. Steps 1–2 need internet but not a GPU.

## 1. Clone and verify

```bash
git clone https://github.com/secureswim/gigpo.git
cd gigpo
python3 week1/check_baseline_scripts.py     # must report the pinned commit and 20 matching settings
```

Do not edit `upstream-verl-agent/`. All changes live in `extension/` and are applied to a copy.

## 2. Install everything

```bash
bash setup/install.sh
```

This creates the conda env `verl-agent-gigpo` (Python 3.12, torch 2.6.0/cu124, flash-attn, vllm 0.8.2, ALFWorld packages), builds `work/verl-agent-ext`, installs it editable, downloads ALFWorld data and Qwen2.5-1.5B-Instruct, and writes `setup/env_lock.txt`. Commit that lock file. If a package needs a different version on this machine, note it in `knowledge_base/status.md` and keep it the same for every run.

At the end it prints the exports to add to your shell profile:

```bash
conda activate verl-agent-gigpo
export ALFWORLD_DATA=$HOME/.cache/alfworld
export MODEL_PATH=$HOME/models/Qwen2.5-1.5B-Instruct
```

## 3. Preflight

```bash
bash setup/preflight.sh
```

This checks the snapshot, the CPU tests, CUDA, vLLM, flash-attn, that `verl` resolves to the patched tree, ALFWorld reset/step, the model and disk space. Every line must say PASS.

## 4. Smoke run (2 steps plus resume)

```bash
N_GPUS=2 bash week1/smoke_gigpo.sh
```

It must end with `SMOKE PASSED`. Use `N_GPUS=1` on a single GPU, and add `MICRO_BS=16` (or 8) if you run out of memory. Keep whichever settings work for every later run.

## 5. Hand back the evidence

Share `runs/preflight.txt`, `runs/gigpo_seed0_smoke/{launches.txt,train.log}`, the `check_smoke.py` output and any install deviations. Then start the run list in `knowledge_base/compute_and_team.md`.
