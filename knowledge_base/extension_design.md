# Anchor-group extension design

## Hypothesis

Some step-level anchor groups give a noisy micro advantage. Removing or down-weighting them, while keeping GiGPO's episode-level (macro) advantage intact, may improve sample efficiency or stability.

## What the original code already does (verified in `gigpo/core_gigpo.py`)

- The micro advantage is the group's discounted return, centred on the group mean and, with `mean_std_norm`, divided by the group std.
- **Exactly degenerate groups already contribute nothing.** A group of size 1, or a group whose returns are all equal, gets a micro advantage of 0. So "filter the groups with zero signal" is a no-op. `tests/test_gigpo_ext.py::test_degenerate_groups_already_get_zero_micro_advantage` demonstrates this.
- **The normalised advantage is scale-free.** A size-2 group always gets ±1/√2, whether its return gap is 0.001 or 10 (`test_std_normalisation_is_scale_free_for_size_two_groups`). Small groups and near-zero-spread groups therefore push the policy as hard as informative ones. These are the groups the extension targets.
- **Caveat:** a small return gap is not automatically noise. With γ = 0.95 and a terminal reward of 10, two successful trajectories that differ by one step have returns about 5% apart. That is real "be faster" signal. Whether removing such groups helps is the empirical question.
- Identical actions do not imply equal returns, and action strings are not reliably available in the batch, so groups are judged by their computed return signal, not by action diversity.

## Diagnostic (runs automatically in every GiGPO-family run)

The patched trainer logs `gigpo_diag/*` metrics every training step and appends the per-group arrays (size, return std, range, mean, applied weight) to `<run>/gigpo_diag/step_groups.jsonl`. The key metrics are:

| Metric | Meaning |
| --- | --- |
| `frac_groups_size1`, `frac_groups_size2`, `frac_groups_size_ge3` | group-size distribution |
| `frac_groups_zero_signal`, `frac_samples_zero_signal` | groups or step samples whose micro advantage is exactly 0 |
| `return_std_p10 … p90` | return spread among groups with spread |
| `frac_spread_groups_std_lt_{0.1,0.5,1.0}` | fixed reference levels, comparable across runs |
| `mean_abs_step_over_episode` | size of the micro advantage relative to the macro one |
| `frac_episode_groups_all_equal` | what DAPO-style dynamic sampling would drop (episode level) |
| `frac_samples_micro_kept`, `micro_mass_ratio`, `weight_scale` | how much signal a filter removed or rescaled |

## Conditions

All are implemented in `extension/gigpo_ext.py` and selected with `algorithm.gigpo.group_filter`.

1. **`none`: original GiGPO.** Uses the upstream functions unchanged. Bit-identical to upstream (`test_filter_none_is_identical_to_upstream`).
2. **`threshold`: condition (b).** The micro advantage is set to 0 for groups with `n < MIN_GROUP_SIZE` or `std < MIN_RETURN_STD`. The macro advantage is unchanged.
3. **`weighted`: condition (c), the "divergence weight".** Divergence means the spread of outcomes after the shared state. Each group's micro advantage is multiplied by

   w_g = s_g / (s_g + τ) · (n_g − 1) / n_g

   where s_g is the group's return std, n_g its size, and τ = `WEIGHT_TAU`. All weights are then multiplied by one batch-wide constant so that Σ|micro advantage| is unchanged. Only the distribution of micro advantage across groups changes, not the overall update size. The (n − 1)/n factor discounts the std estimate of tiny groups.

## Predeclared threshold rule

Thresholds are set from **training batches of the original-GiGPO control run**, steps 1–20, by `analysis/choose_thresholds.py`. Validation results are never used.

- "Spread groups" are groups with n ≥ 2 and std > 1e-6.
- `MIN_GROUP_SIZE = 3`.
- `MIN_RETURN_STD` = 25th percentile of spread-group std.
- `WEIGHT_TAU` = median of spread-group std.

**Null-result gate:** if the threshold filter would touch fewer than 5% of the step samples that currently receive a non-zero micro advantage, report the diagnostic as a null result and do not train (b) and (c).

## Relationship to existing methods

- **Dynamic sampling (DAPO) is already in the pinned snapshot**, not only in newer upstream code. `examples/gigpo_dynamic_trainer/run_alfworld.sh` sets `algorithm.filter_groups.enable=True`, and `agent_system/multi_turn_rollout/rollout_loop.py` resamples **episode-level** groups whose episodes all got the same reward. Its script also turns on clip-higher (`clip_ratio_high=0.28`), so it is not a pure filter comparison. It can be run as an optional reference (`gigpo_dynamic`). It spends extra environment steps on resampling, which `env/steps_collected` counts. Our conditions act on **step-level groups after anchor matching** and spend no extra rollouts.
- **Similarity-based GiGPO is not in the pinned snapshot.** In current upstream `main` (checked at commit `20bd331`, 9 June 2026), `algorithm.gigpo.enable_similarity` and `similarity_thresh` (default 0.95) merge anchor observations whose `difflib.SequenceMatcher` ratio is ≥ the threshold, using a greedy first-match clustering. That changes **which samples form a group**. We change **how a formed group is used**. The two axes are independent. Cite it as related work and do not mix it into the pinned baseline.

## Decision rule

If the gate fails, or filtering gives no gain beyond seed-to-seed variation, report the null result. If it helps, check that the gain survives a second seed and repeated evaluation, and report environment steps to threshold, not only final success.
