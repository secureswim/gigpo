# Anchor-group extension plan

## Hypothesis

Some step-level anchor groups provide little useful relative-advantage signal. The project tests whether filtering or down-weighting such groups improves sample efficiency or stability while keeping episode-level GiGPO credit assignment intact.

## Diagnostic before implementation

Instrument the group formation and step-advantage path in `gigpo/core_gigpo.py`. For each group, record its size, discounted-return variance, mean absolute step advantage, and distinct action count if the actions can be aligned reliably. Aggregate distributions by training step and task type. First verify which fields are available in the trainer batch; do not add an action-diversity metric by guessing from token strings.

Define a candidate degenerate group by **computed signal**, for example group size one or near-zero return variance. Inspect the actual scale before choosing a threshold. Identical actions may still have different returns, so do not equate them with zero signal. Preserve the episode-level advantage for filtered groups.

## Conditions

1. Original pinned GiGPO.
2. Variance-filtered step groups: set only the micro-advantage contribution of groups below the fixed signal threshold to zero.
3. Divergence-weighted step groups: scale the micro-advantage contribution by a predeclared function of group signal, with normalization that prevents changing the overall update magnitude unintentionally.

Start with condition 2. Add condition 3 only after the diagnostic and filtered comparison are sound. Use the same compute budget, evaluation procedure, and training seeds for all conditions. Check how much training signal each method removes or scales, not just final success.

## Relationship to existing methods

The newer upstream repository includes dynamic sampling for GiGPO, following DAPO. That samples or rejects **episode-level rollout groups**. This proposal focuses on step-level groups **after anchor matching**. Keep those axes distinct in the report and inspect the newer code before making a novelty claim.

## Decision rule

If near-zero-signal groups are rare, or their removal has no measurable benefit, report the null result. If filtering helps, examine whether the gain survives repeated evaluations and whether it changes environment steps consumed. Do not use the final validation set to pick the threshold after observing results.
