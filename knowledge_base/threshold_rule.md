# How the extension thresholds are chosen (rule v2)

**Status:** adopted 18 November 2026, before any full training run (`results/decisions.md`). It replaces rule v1, which was never used. The rule is implemented in `analysis/choose_thresholds.py` and checked by `tests/test_analysis.py`.

Background for this page: [`reward_and_return_structure.md`](reward_and_return_structure.md) (what a group's return spread can mean) and [`extension_design.md`](extension_design.md) (what the two extensions do).

## The measure

Both extensions act on each anchor group's **return spread `s_g`**: the unbiased standard deviation of the discounted step returns of the samples in the group. They also use the group size `n_g`. These are the same quantities GiGPO's own `step_norm_reward` uses.

- `gigpo_threshold` sets the step (micro) advantage of a group to 0 when `n_g < MIN_GROUP_SIZE` or `s_g < MIN_RETURN_STD`.
- `gigpo_weighted` multiplies it by `w_g = s_g/(s_g+τ) · (n_g−1)/n_g`, with τ = `WEIGHT_TAU`, then rescales all weights by one batch constant so the total micro-advantage mass is unchanged.

The episode (macro) advantage is never changed.

## Rule v2

| Value | Rule v2 | Rule v1 (retired) |
| --- | --- | --- |
| `MIN_GROUP_SIZE` | **2**: no size filter; size-1 groups already get 0 | 3 |
| `MIN_RETURN_STD` | **0.075, fixed** | 25th percentile of `s_g`, steps 1–20 |
| `WEIGHT_TAU` | median of `s_g` over spread groups, control steps 1–20 | same |

"Spread groups" are groups with n ≥ 2 and `s_g` > 1e-6. Only training batches of the `gigpo` control are used, never validation. The fallback is unchanged: if steps 1–20 have no spread group, retry with `--steps <approved-final-step>`; if there is still none, write no thresholds and go to H3.

## Why v1 was retired

1. **Its percentile had no fixed meaning.** `s_g` is multimodal: penalty-only groups sit at ≤ 0.0707, success-vs-failure groups at about 0.5–7, timing differences in between. p25 would fall in whichever band the early-training mixture happened to put it. It could remove exactly the "avoid invalid actions" signal, or split the efficiency band, with no stated hypothesis behind either.
2. **`MIN_GROUP_SIZE = 3` removed every size-2 group, including success-vs-failure pairs.** Those are the clearest step-level signal there is. The motivation (a size-2 group always gets ±1/√2) is about the *size* of the push, but the *direction* is correct. On one synthetic dataset with the real return structure, v1 would have removed 31.9% of micro-advantage samples, against 13.0% for v2. The real report prints both shares for the control.
3. **The rule was a step beyond DAPO without DAPO's justification.** DAPO only drops groups whose advantage is exactly zero ([`dapo_and_related_work.md`](dapo_and_related_work.md)).

## Why 0.075

A group whose returns span at most one invalid-action penalty (range ≤ 0.1) has `s_g ≤ 0.1/√2 = 0.0707`. 0.075 sits just above that, with room for float32 rounding of the returns. The threshold condition therefore tests one stated hypothesis:

> Groups whose returns differ by no more than one invalid-action penalty carry too little signal to deserve a full-size, std-normalised step advantage.

What it removes, exactly:
- **Always:** groups that never win and differ only by an invalid action, and groups that all win at the same distance with one invalid action.
- **Also:** success-only groups whose timing gap is ≤ ~0.1, which happens about 31 or more steps from the goal, and large groups with one slightly different member. The report states this share separately ("penalty-scale groups that are success-only").
- **Never:** any group containing both a later win and a non-win. Its range is ≥ 0.71, so `s_g` ≥ 0.2 even for size 8 with one outlier.

Since the value is derived from the reward structure, it needs no data and cannot drift with the success rate. `WEIGHT_TAU` remains data-derived because the weighted variant is meant to be relative to the spreads the control actually produced.

## What to expect from each condition (a design expectation, not a result)

On synthetic data built from the real return structure (`results/` holds no real data yet):
- **`gigpo_threshold`** removes the penalty-scale groups only, about 10–30% of the step samples that carry a micro advantage. That share shrinks as the success rate rises, because success-vs-failure groups become more common.
- **`gigpo_weighted`** with τ = median keeps success-vs-failure groups near the top weight. It strongly down-weights timing groups (raw weight ≈ 0.08 vs ≈ 0.4) and penalty groups (≈ 0.03). After mass rescaling, it **concentrates the micro advantage on success-vs-failure groups**. This makes it a sharper intervention than thresholding, and it suppresses the "be faster" signal as well. Keep that in mind when interpreting episode-length curves.

Check these against the real `results/diagnostic/diagnostic_report.md` once it exists. The report's band table shows them directly.

## What the diagnostic report contains

`analysis/choose_thresholds.py` writes `results/diagnostic/diagnostic_report.md`, `thresholds.env` and `anchor_groups.png`:
- **Group overview:** size distribution, share of groups with equal returns, share of step samples already at zero micro advantage, and std percentiles.
- **Thresholds and the rule version.**
- **Band table:** spread groups split into *mixed outcome*, *timing* (success-only, `s_g` ≥ 0.075), *penalty-scale* (`s_g` < 0.075) and *other* (expected empty; if not, report it). For each band it gives the share of groups, the share of step samples, the median std and the mean weighted-GiGPO raw weight.
- **Penalty-scale composition** (success-only vs never-win).
- **Band shares by 5-step block,** to show drift within steps 1–N. Report a drift; do not re-derive thresholds.
- **Threshold impact forecast:** the share of micro-advantage samples the threshold removes, with the 5% low-impact marker. Either way, both extensions run.
- **The retired v1 share,** for the write-up only.
- **Figure:** group-size distribution, and a log-scale `s_g` histogram stacked by band with `MIN_RETURN_STD` and `WEIGHT_TAU` marked.

Bands need the per-group `mins`/`maxs` that `extension/gigpo_ext.py` records (added with rule v2). A record without them gets only the std split, and the report says so.

## Changing the rule

The rule is part of the experiment. It may change only with written team approval recorded in `results/decisions.md`, and only before the control's diagnostic is read. An agent operating the runbook must never change it (`AGENTS.md` rule 3).
