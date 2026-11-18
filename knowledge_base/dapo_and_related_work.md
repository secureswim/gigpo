# DAPO and related work: what transfers to step-level GiGPO

The two extensions were inspired by DAPO's dynamic sampling. This page records what DAPO actually does, what carries over to GiGPO's anchor groups, and what does not. It explains why rule v2 ([`threshold_rule.md`](threshold_rule.md)) looks the way it does.

## DAPO in brief

Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) (ByteDance Seed and Tsinghua AIR, March 2025). DAPO is GRPO with four changes, trained on math reasoning with long chain-of-thought:

1. **Clip-Higher.** Decoupled PPO clip bounds, ε_low = 0.2 and ε_high = 0.28. A higher upper bound lets low-probability tokens gain probability, which counters entropy collapse.
2. **Dynamic Sampling.** Prompts whose sampled answers are all correct or all wrong give every response the same reward, so the group-normalised advantage is exactly 0 and contributes no gradient. DAPO drops those groups and keeps sampling until the batch is full of groups with nonzero advantage. The batch keeps a constant number of informative prompts.
3. **Token-level policy-gradient loss.** The loss is averaged over all tokens in the batch rather than per sample then per batch, so long responses are not under-weighted.
4. **Overlong reward shaping.** Truncated responses are filtered or softly penalised, which removes reward noise.

DAPO also drops the KL penalty. With Qwen2.5-32B it reports 50 points on AIME 2024, beating DeepSeek-R1-Zero-Qwen-32B (47) with half the training steps. In the paper's ablation, dynamic sampling was the largest single step. Recheck the exact ablation numbers in the paper before citing them; this page gives only the headline result.

## What transfers to our setting, and what does not

| DAPO idea | Status in this repository |
| --- | --- |
| Drop zero-advantage groups (episode level) | **Already in the pinned snapshot**: `examples/gigpo_dynamic_trainer/run_alfworld.sh` (`algorithm.filter_groups.enable=True`) resamples episode groups whose episodes all got the same reward. That script also sets `clip_ratio_high=0.28`. It is **outside this handoff**. |
| Drop zero-advantage groups (step level) | **A no-op.** GiGPO already gives size-1 and equal-return anchor groups exactly 0 micro advantage. Also, those samples still carry the episode advantage, so they are not zero-gradient samples the way DAPO's are. |
| Resample to keep the batch full of informative groups | **Not done.** Our extensions spend no extra rollouts, so the comparison is at a fixed number of environment steps per training step. |
| Drop *low-but-nonzero* spread groups | **Not in DAPO.** DAPO's filter has no tuned threshold and treats every nonzero-variance group as informative. With binary rewards, every such group is. `gigpo_threshold` and `gigpo_weighted` go beyond DAPO here, which is why the threshold must come from a stated hypothesis rather than a percentile. |
| Clip-Higher, token-level loss, overlong shaping | Not used; they would change the training objective for all conditions. |

**Bottom line:** the extensions are DAPO-*inspired* (decide which groups deserve gradient) but are not a DAPO port. DAPO's evidence supports removing groups with *no* signal. It does not by itself say that removing *weak* signal helps. That is the open question this project tests.

## Dr. GRPO: the other side of the same problem

Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) (March 2025). The paper argues that GRPO's division by the per-group reward std biases learning toward groups with low reward variance, and that length normalisation biases response length. Dr. GRPO removes both terms.

This is the root of the problem our extensions address. Under `mean_std_norm`, a size-2 anchor group whose returns differ by 0.1 gets the same ±0.707 as one that differs by 10. Upstream's alternative mode `mean_norm` (centring without dividing by std) would make the micro advantage proportional to the gap, which is the Dr. GRPO-style fix. We cannot use it here: `algorithm.gigpo.mode` is a protected setting, and the paper's 86.7% result uses `mean_std_norm`. `gigpo_weighted` can be read as a partial, mass-preserving move toward gap-proportional advantages within `mean_std_norm`. Mention this in the write-up.

## Other related work already noted

- **Similarity-based GiGPO** (current upstream `main`, not the pinned snapshot): merges near-identical anchor observations (`difflib` ratio ≥ 0.95). It changes which samples form a group; we change how a formed group is used. The two are independent. See `extension_design.md`.
- **GiGPO itself:** Feng et al., [arXiv:2505.10978](https://arxiv.org/abs/2505.10978), NeurIPS 2025.
