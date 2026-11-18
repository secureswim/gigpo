# Knowledge base

Background for anyone, person or agent, who operates or analyses the seed-0 GiGPO study. **The procedure itself lives in [`../HANDOFF_SEED0.md`](../HANDOFF_SEED0.md) and [`../RUNBOOK_OMNI.md`](../RUNBOOK_OMNI.md); the rules for agents in [`../AGENTS.md`](../AGENTS.md).** Nothing here overrides them.

## Reading order

| # | File | Read it to learn | Needed for |
| --- | --- | --- | --- |
| 1 | [`status.md`](status.md) | where the project stands, what is done, every decision, known uncertainties | everyone, first |
| 2 | [`operator_faq.md`](operator_faq.md) | short answers to the questions an operator usually has, with pointers | the GPU operator |
| 3 | [`experiment_protocol.md`](experiment_protocol.md) | what is compared, what is fixed, the outcome measures | operator and analyst |
| 4 | [`method_and_code.md`](method_and_code.md) | how GiGPO computes advantages, the upstream file map, metric names | debugging, analysis |
| 5 | [`extension_design.md`](extension_design.md) | the two extensions, the diagnostics, the decision rule | analysis |
| 6 | [`reward_and_return_structure.md`](reward_and_return_structure.md) | the exact reward and return values in text ALFWorld, and what a group's return spread can mean | reading the diagnostic report |
| 7 | [`threshold_rule.md`](threshold_rule.md) | rule v2 for `MIN_GROUP_SIZE`, `MIN_RETURN_STD` and `WEIGHT_TAU`, why v1 was retired, and how to read the report | runbook step 8 and the write-up |
| 8 | [`dapo_and_related_work.md`](dapo_and_related_work.md) | DAPO, Dr. GRPO and how they relate to the extensions | the write-up |
| 9 | [`compute_and_team.md`](compute_and_team.md) | run list, budget formula, hardware, work split | H2 budgeting |
| — | [`archive/`](archive/) | superseded notes (IIIT `gpurun` server); **do not follow** | history only |

## Glossary

| Term | Meaning |
| --- | --- |
| **Anchor observation / anchor group** | Step samples from the 8 rollouts of the same task (`index`) that saw an identical observation. GiGPO compares their returns. One trajectory that revisits a state contributes several members. |
| **Episode (macro) advantage** | GRPO-style advantage of the whole episode relative to the other rollouts of the same task. Never changed by our extensions. |
| **Step (micro) advantage** | An anchor-group member's return centred on the group mean and, under `mean_std_norm`, divided by the group std. Added to the macro advantage with weight ω = 1. |
| **Return spread `s_g`** | Unbiased std of the discounted step returns in a group. The measure the extensions act on. |
| **Spread group** | Group with ≥ 2 members and `s_g` > 1e-6. Only these carry a nonzero micro advantage. |
| **Bands** | Diagnostic classes of spread groups. *Mixed*: some members' trajectories later win, others do not. *Timing*: all later win, `s_g` ≥ 0.075. *Penalty-scale*: `s_g` < 0.075, removed by `gigpo_threshold`. |
| **`MIN_GROUP_SIZE`, `MIN_RETURN_STD`, `WEIGHT_TAU`** | Extension settings written to `results/diagnostic/thresholds.env` by `analysis/choose_thresholds.py` under rule v2: 2, 0.075 (fixed) and the median `s_g` of control steps 1–20. |
| **Micro mass** | Σ\|step advantage\| over the batch. `gigpo_weighted` rescales its weights to keep it unchanged. |
| **Control** | `gigpo_seed0`, original GiGPO. Its training diagnostics set the thresholds. |
| **H1 / H2 / H3** | Human approval gates: resources, budget, anything unplanned. Record each in `results/decisions.md`. |
| **Hardware knobs** | `N_GPUS`, `MICRO_BS`, `GPU_MEM_UTIL`, `VAL_BATCH`, `REF_OFFLOAD` in `cluster/omni/resources.env`. The only settings that may change, and only by the runbook ladder. |
| **Patched tree** | `work/verl-agent-ext`, a copy of `upstream-verl-agent/` plus `extension/`. Built by `prepare.sh`, never edited by hand. |
| **Credits** | Omni billing unit. Price and balance are found in runbook step 3. |

## Where the evidence is

| Question | Source |
| --- | --- |
| What did each run do? | `results/<condition>_seed0.md` (from `results/TEMPLATE.md`) and `runs/<run>/launches.txt` |
| Who approved what? | `results/decisions.md` |
| What did the thresholds come from? | `results/diagnostic/diagnostic_report.md` and `thresholds.env` |
| What was the first GPU evidence? | `results/omni-short-10333/README.md` (reduced-settings smoke, pre-extension code) |
| Is the snapshot pristine? | `python3 week1/check_baseline_scripts.py` and `UPSTREAM_SHA256SUMS.txt` |
| Does the code still behave? | `python3 -m pytest -q tests` (CPU only) |
