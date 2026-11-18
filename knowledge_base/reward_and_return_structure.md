# Reward, return and anchor-group structure (text ALFWorld, pinned code)

This page derives, from the pinned code, what values a step return can take and therefore what the return spread `s_g` of an anchor group can mean. Rule v2 of the threshold choice ([`threshold_rule.md`](threshold_rule.md)) rests on it. Every claim names the file it comes from. Paths are relative to `upstream-verl-agent/`.

## 1. Reward

`agent_system/environments/env_package/alfworld/envs.py::compute_reward`, text mode (`multi_modal=False`):

```python
reward = 10.0 * float(info['won'])
```

So the only environment reward is **10 on the step that wins the task**, 0 otherwise. `goal_condition_success_rate` is used only in the multimodal branch, which we do not run.

## 2. Discounted step return

`gigpo/core_gigpo.py::compute_step_discounted_returns` is called in `verl/trainer/ppo/ray_trainer.py` (`fit`, around line 929) with `algorithm.gamma = 0.95`:

```
G_t = r_t + 0.95 * G_{t+1}
```

With one reward of 10 at the winning step, a step that is k steps before the win has `G = 10 · 0.95^k`. A step on a trajectory that never wins has `G = 0`. `env.max_steps = 50`, so k ≤ 49 and the smallest positive return is `10 · 0.95^49 ≈ 0.81`.

## 3. Invalid-action penalty

`ray_trainer.py::apply_invalid_action_penalty` runs **after** the returns are computed (around line 979). It subtracts `invalid_action_penalty_coef = 0.1` from `step_rewards[i]` for that step only:

```python
step_rewards[i] -= invalid_action_penalty_coef * action_invalids
```

The penalty is therefore **not discounted or propagated** to earlier steps. It shifts only the return of the step whose own action was invalid. (It is also subtracted from that step's token-level score, which feeds the episode-level advantage.)

**Possible step-return values:** `10 · 0.95^k` or `10 · 0.95^k − 0.1` on a trajectory that later wins (≥ 0.71 in every case), and `0` or `−0.1` on one that never wins. A threshold at any value between 0 and 0.71 separates the two cleanly. The code uses `SUCCESS_SPLIT = 0.3`.

## 4. Anchor groups

`core_gigpo.py::build_step_group` groups step samples that share the episode group `index` (the same task; 8 rollouts per task) **and** an identical `anchor_obs`. It does not distinguish trajectories, so **one trajectory that revisits a state forms a group with itself**. In the reduced smoke run (job 10333) the mean group size was 9.3, more than the 8 rollouts, which shows that revisits happen.

`step_norm_reward` centres each return on its group mean. With `mode = mean_std_norm` (the paper's "w/ std" setting) it divides by the group's unbiased std plus 1e-6. Size-1 groups and groups with equal returns get exactly 0.

## 5. What a group's spread can mean

For a two-valued group with gap d, the sample std is `d · sqrt(p(1−p) · n/(n−1))`, which is at most `d/√2` (reached at n = 2). Any group whose returns span a range ≤ d has std ≤ d/√2.

| What differs inside the group | Example (size 2) | `s_g` |
| --- | --- | --- |
| Only the invalid-action penalty (both never win, or both win at the same k) | 0 vs −0.1 | 0.1/√2 = **0.0707** (smaller for larger groups) |
| Both win, one step apart, k steps from the goal | 10·0.95^k vs 10·0.95^(k+1) | 0.354·0.95^k: **0.35** at k=0, **0.21** at k=10, **0.077** at k=30, **0.04** at k=49 |
| One wins k steps later, one never wins | 10·0.95^k vs 0 | 7.07·0.95^k: **7.07** at k=0, **4.2** at k=10, **0.57** at k=49 |
| A revisit inside one winning trajectory, j steps apart | 10·0.95^k vs 10·0.95^(k+j) | timing-like; the earlier visit gets the lower return |
| A revisit inside one failing trajectory | 0 vs 0 (or −0.1) | 0 or penalty-scale |

Consequences:

1. **`s_g` is multimodal.** Penalty-only groups pile up at or below 0.0707. Success-vs-failure groups sit around 0.5–7. Timing differences fall in between, and far from the goal (k ≳ 31) they drop into the penalty band.
2. **A percentile of `s_g` has no fixed meaning.** It lands wherever the mixture of these bands puts it, and the mixture shifts as the success rate rises. That is why rule v1 (p25) was retired.
3. **Under `mean_std_norm` every size-2 group gets ±1/√2 ≈ 0.707**, whether its gap is 0.1 or 10. A penalty-only pair pushes the policy as hard as a success-vs-failure pair. The *sign* is still right in both cases: the invalid action gets the negative advantage.
4. **ALFWorld text is deterministic given the actions**, so no return difference is measurement noise. A small gap is a small but real consequence of the choices made after the shared state.

## 6. Where these constants are checked

`analysis/choose_thresholds.py` hard-codes `INVALID_ACTION_PENALTY = 0.1`, `WIN_REWARD = 10`, `GAMMA = 0.95` and `MAX_STEPS = 50`. `tests/test_analysis.py::test_rule_constants_match_upstream` fails if the pinned upstream script or `envs.py` says otherwise.
