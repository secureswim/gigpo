"""Anchor-group diagnostic and the predeclared rule (v2) that sets extension thresholds.

    python3 analysis/choose_thresholds.py --diag runs/gigpo_seed0/gigpo_diag/step_groups.jsonl

Reads the per-group arrays that the original-GiGPO control run writes every
training step. Uses TRAINING batches only (never validation results), from steps
1..--steps (default 20), so thresholds are fixed before any extension run.

Predeclared rule v2 (adopted 18 Nov 2026, before any full run; rationale in
knowledge_base/threshold_rule.md and knowledge_base/reward_and_return_structure.md):

  * "spread groups"  = anchor groups with >= 2 members and return std > 1e-6
  * MIN_GROUP_SIZE   = 2      no size filter beyond size-1 groups, which already
                              get zero micro advantage. (v1 used 3, which also
                              removed every size-2 success-vs-failure pair.)
  * MIN_RETURN_STD   = 0.075  FIXED, not estimated. In text ALFWorld a step return
                              is 10*0.95^k (the trajectory later wins) or 0, minus
                              0.1 if this step's own action was invalid. Any group
                              whose returns span at most one invalid-action penalty
                              (range <= 0.1) has sample std <= 0.1/sqrt(2) = 0.0707.
                              0.075 sits just above that, so the threshold condition
                              tests one stated hypothesis: "groups whose returns
                              differ by no more than one invalid-action penalty
                              carry too little signal to deserve a full-size,
                              std-normalised step advantage".
  * WEIGHT_TAU       = median of spread-group std (unchanged from v1)
  * A <5% touched share predicts a small threshold effect; it does not stop
    either extension run, since weighted GiGPO can still reweight other groups.
  * If steps 1..20 contain no spread groups, retry after the control finishes
    with --steps <approved-final-step>. If none exist then, report the mechanism
    result and do not fabricate threshold values.

The report also classifies spread groups into return bands (mixed outcome,
success-only timing differences, penalty-scale) so the write-up can state what
each extension actually changed. Classification needs the per-group min/max that
extension/gigpo_ext.py records; older records without them get std bands only.

Writes <out>/thresholds.env (cluster/omni/job.sbatch sources it for the
threshold and weighted conditions), <out>/diagnostic_report.md and
<out>/anchor_groups.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RULE_VERSION = "v2"
ZERO_STD = 1e-6
# Reward structure of the pinned text-ALFWorld setting (checked against the
# upstream script by tests/test_analysis.py::test_rule_constants_match_upstream).
INVALID_ACTION_PENALTY = 0.1            # actor_rollout_ref.actor.invalid_action_penalty_coef
WIN_REWARD = 10.0                       # envs.py: reward = 10 * won (text mode)
GAMMA = 0.95                            # algorithm.gamma
MAX_STEPS = 50                          # env.max_steps
PENALTY_ONLY_STD = INVALID_ACTION_PENALTY / np.sqrt(2)   # 0.0707: max std when range <= penalty
MIN_GROUP_SIZE = 2
MIN_RETURN_STD = 0.075
# Smallest return of a step whose trajectory later wins: 10*0.95^49 - 0.1 = 0.71.
# Steps whose trajectory never wins have return 0 or -0.1. Anything between
# separates the two cleanly.
SUCCESS_SPLIT = 0.3
LOW_IMPACT_MARK = 0.05

BANDS = ("mixed", "timing", "penalty", "other")
BAND_LABEL = {
    "mixed": "Mixed outcome (some later win, some do not)",
    "timing": "All later win, returns differ by timing (std >= MIN_RETURN_STD)",
    "penalty": "Penalty-scale spread (std < MIN_RETURN_STD) -- removed by gigpo_threshold",
    "other": "Other (std >= MIN_RETURN_STD, no win; not expected, check data)",
}
# Categorical slots 1-4 of the default validated palette, in fixed order.
BAND_COLOR = {"mixed": "#2a78d6", "timing": "#eb6834", "penalty": "#1baf7a", "other": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(path, max_step):
    steps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if 1 <= rec["step"] <= max_step:
                steps.append(rec)
    if not steps:
        raise SystemExit(f"No training steps 1..{max_step} in {path}")
    return steps


def classify(stds, mins, maxs):
    """Band of each group (only meaningful for spread groups)."""
    band = np.full(len(stds), "other", dtype=object)
    low = stds < MIN_RETURN_STD
    if mins is None:
        band[low] = "penalty"
        band[~low] = "unclassified"
        return band
    win_any = maxs > SUCCESS_SPLIT
    win_all = mins > SUCCESS_SPLIT
    band[win_any & ~win_all] = "mixed"
    band[win_all & ~low] = "timing"
    band[low & ~(win_any & ~win_all)] = "penalty"
    return band


def weighted_raw(sizes, stds, tau):
    return stds / (stds + tau) * (sizes - 1) / np.maximum(sizes, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True, help="step_groups.jsonl from the original GiGPO run")
    ap.add_argument("--steps", type=int, default=20, help="use training steps 1..N")
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()

    steps = load(args.diag, args.steps)
    has_minmax = all("mins" in s and "maxs" in s for s in steps)
    sizes = np.concatenate([np.array(s["sizes"]) for s in steps])
    stds = np.concatenate([np.array(s["stds"], dtype=float) for s in steps])
    step_of = np.concatenate([np.full(len(s["sizes"]), s["step"]) for s in steps])
    mins = np.concatenate([np.array(s["mins"], dtype=float) for s in steps]) if has_minmax else None
    maxs = np.concatenate([np.array(s["maxs"], dtype=float) for s in steps]) if has_minmax else None
    multi = sizes >= 2
    spread = multi & (stds > ZERO_STD)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not spread.any():
        (out / "thresholds.env").unlink(missing_ok=True)
        msg = (f"No anchor group with nonzero return spread in training steps 1..{args.steps}. "
               "No extension thresholds can be derived. If this was the 20-step check, "
               "finish the control and retry with --steps <approved-final-step>. "
               "If this was the full-run check, report the mechanism result and "
               "request a new experiment plan.")
        (out / "diagnostic_report.md").write_text("# Anchor-group diagnostic\n\n" + msg + "\n", encoding="utf-8")
        raise SystemExit(msg)

    s_sizes, s_stds, s_steps = sizes[spread], stds[spread], step_of[spread]
    tau = float(np.median(s_stds))
    band = classify(s_stds, mins[spread] if has_minmax else None, maxs[spread] if has_minmax else None)

    n_samples = sizes.sum()
    signal_samples = s_sizes.sum()
    zero_share = 1 - signal_samples / n_samples
    touched = s_stds < MIN_RETURN_STD      # MIN_GROUP_SIZE=2 removes nothing extra
    touched_share = s_sizes[touched].sum() / signal_samples
    low_impact = touched_share < LOW_IMPACT_MARK

    # Retired rule v1, for transparency only (never written to thresholds.env).
    v1_std = float(np.percentile(s_stds, 25))
    v1_touched = (s_sizes < 3) | (s_stds < v1_std)
    v1_share = s_sizes[v1_touched].sum() / signal_samples

    (out / "thresholds.env").write_text(
        f"# rule {RULE_VERSION} (analysis/choose_thresholds.py); from {args.diag}, "
        f"training steps 1..{args.steps} ({len(steps)} steps)\n"
        f"export MIN_GROUP_SIZE={MIN_GROUP_SIZE}\n"
        f"export MIN_RETURN_STD={MIN_RETURN_STD:g}\n"
        f"export WEIGHT_TAU={tau:.6g}\n", encoding="utf-8")

    size_counts = {k: int((sizes == k).sum()) for k in range(1, int(sizes.max()) + 1) if (sizes == k).any()}
    present = [b for b in BANDS if (band == b).any()] + (["unclassified"] if (band == "unclassified").any() else [])
    w_raw = weighted_raw(s_sizes, s_stds, tau)

    def pct(x):
        return f"{x:.1%}"

    lines = [
        "# Anchor-group diagnostic",
        "",
        f"Source: `{args.diag}`, training steps 1..{args.steps} ({len(steps)} steps found). Rule {RULE_VERSION}.",
        "",
        "## Group overview",
        "",
        f"- Anchor groups: {len(sizes)}; step samples: {int(n_samples)}",
        f"- Groups of size 1: {pct(np.mean(sizes == 1))}; size 2: {pct(np.mean(sizes == 2))}; size >= 3: {pct(np.mean(sizes >= 3))}",
        f"- Groups with >= 2 members but equal returns: {pct(np.mean(multi & ~spread))}",
        f"- Step samples with zero micro advantage already (size 1 or equal returns): {pct(zero_share)}",
        f"- Spread groups: {int(spread.sum())} holding {int(signal_samples)} step samples",
        f"- Spread-group return std: p10 {np.percentile(s_stds, 10):.4g}, p25 {v1_std:.4g}, "
        f"median {tau:.4g}, p75 {np.percentile(s_stds, 75):.4g}, p90 {np.percentile(s_stds, 90):.4g}",
        "",
        f"## Thresholds (predeclared rule {RULE_VERSION})",
        "",
        f"- MIN_GROUP_SIZE = {MIN_GROUP_SIZE} (no size filter beyond size-1 groups)",
        f"- MIN_RETURN_STD = {MIN_RETURN_STD:g} (fixed: just above {PENALTY_ONLY_STD:.4f} = "
        f"{INVALID_ACTION_PENALTY:g}/sqrt(2), the largest std of a group whose returns span <= one invalid-action penalty)",
        f"- WEIGHT_TAU = {tau:.6g} (median of spread-group std, steps 1..{args.steps})",
        "",
        "## Return bands of spread groups",
        "",
    ]
    if not has_minmax:
        lines += ["The record has no per-group min/max (written by an older extension), so only the "
                  "std split is shown.", ""]
    lines += ["| Band | Groups | Step samples | Median std | Mean weighted-GiGPO raw weight |",
              "| --- | --- | --- | --- | --- |"]
    for b in present:
        m = band == b
        lines.append(f"| {BAND_LABEL.get(b, b)} | {pct(m.mean())} | {pct(s_sizes[m].sum() / signal_samples)} | "
                     f"{np.median(s_stds[m]):.4g} | {w_raw[m].mean():.3f} |")
    lines += ["",
              "Percentages are of spread groups and of their step samples. The raw weight is "
              "s/(s+tau)*(n-1)/n before the batch-wide rescaling that keeps total micro mass fixed.",
              ""]
    if has_minmax:
        pen = band == "penalty"
        if pen.any():
            pen_win = pen & (mins[spread] > SUCCESS_SPLIT)
            lines += [f"Of the penalty-scale groups, {pct(pen_win.sum() / pen.sum())} are success-only "
                      "(timing differences far from the goal or one invalid action); the rest never win "
                      "and differ only by the invalid-action penalty.", ""]

    blocks = sorted(set(((s_steps - 1) // 5).tolist()))
    if len(blocks) > 1:
        lines += ["### Band shares by training-step block", "",
                  "| Steps | " + " | ".join(present) + " |", "| --- |" + " --- |" * len(present)]
        for blk in blocks:
            m = ((s_steps - 1) // 5) == blk
            row = [pct(np.mean(band[m] == b)) for b in present]
            lines.append(f"| {blk * 5 + 1}-{blk * 5 + 5} | " + " | ".join(row) + " |")
        lines += ["", "Shares of spread groups. A drift across blocks means the steps 1..N mix may not "
                  "represent the rest of training; report it, do not re-derive thresholds.", ""]

    lines += [
        "## Threshold impact forecast",
        "",
        f"The threshold filter would remove the micro advantage of {pct(touched_share)} of the step samples "
        f"that currently receive one (low-impact marker: {LOW_IMPACT_MARK:.0%}).",
        "",
        ("**Low threshold impact expected**: still train both extensions; the weighted variant can change other groups."
         if low_impact else "**Noticeable threshold impact expected**: train both extensions."),
        "",
        f"For comparison only, retired rule v1 (MIN_GROUP_SIZE=3, MIN_RETURN_STD=p25={v1_std:.4g}) would have "
        f"removed {pct(v1_share)} of these step samples.",
        "",
        "Group-size counts: " + ", ".join(f"{k}: {v}" for k, v in size_counts.items()),
    ]
    (out / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, gridspec_kw={"width_ratios": [1, 1.5]})
    ks = list(size_counts)
    axes[0].bar(ks, [size_counts[k] / len(sizes) for k in ks], color=BAND_COLOR["mixed"], width=0.7)
    axes[0].set_xlabel("Anchor-group size", color=MUTED)
    axes[0].set_ylabel("Share of groups", color=MUTED)
    lo, hi = max(s_stds.min(), 1e-3), max(s_stds.max(), MIN_RETURN_STD * 2)
    bins = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.1), 41)
    data = [s_stds[band == b] for b in present]
    axes[1].hist(data, bins=bins, stacked=True, color=[BAND_COLOR.get(b, MUTED) for b in present],
                 label=[b for b in present], edgecolor="white", linewidth=0.5)
    axes[1].set_xscale("log")
    for v, name in [(MIN_RETURN_STD, "MIN_RETURN_STD"), (tau, "WEIGHT_TAU")]:
        axes[1].axvline(v, color=INK, linewidth=1, linestyle="--")
        axes[1].annotate(name, (v, axes[1].get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points",
                         fontsize=8, color=INK)
    axes[1].set_xlabel("Return std within group (spread groups, log scale)", color=MUTED)
    axes[1].set_ylabel("Groups", color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes:
        ax.grid(True, axis="y", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "anchor_groups.png")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nWrote {out/'thresholds.env'}, {out/'diagnostic_report.md'}, {out/'anchor_groups.png'}")


if __name__ == "__main__":
    main()
