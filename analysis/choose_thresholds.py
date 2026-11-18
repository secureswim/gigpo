"""Anchor-group diagnostic and the predeclared rule that sets extension thresholds.

    python3 analysis/choose_thresholds.py --diag runs/gigpo_seed0/gigpo_diag/step_groups.jsonl

Reads the per-group arrays that the original-GiGPO control run writes every
training step. Uses TRAINING batches only (never validation results), from steps
1..--steps (default 20), so thresholds are fixed before any extension run.

Predeclared rule (also in knowledge_base/extension_design.md):
  * "spread groups" = groups with >= 2 members and return std > 1e-6
  * MIN_GROUP_SIZE  = 3   (size-2 groups always receive +-1/sqrt(2) under
                           mean_std_norm, whatever their return gap)
  * MIN_RETURN_STD  = 25th percentile of spread-group std
  * WEIGHT_TAU      = median of spread-group std
  * A <5% touched share predicts a small threshold effect; it does not stop
    either extension run, since weighted GiGPO can still reweight other groups.
  * If steps 1..20 contain no spread groups, retry after the control finishes
    with --steps <approved-final-step>. If none exist then, report the mechanism result and do
    not fabricate threshold values.

Writes <out>/thresholds.env (cluster/omni/job.sbatch sources it for the
threshold and weighted conditions), a markdown
report and figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ZERO_STD = 1e-6
MIN_GROUP_SIZE = 3
LOW_IMPACT_MARK = 0.05
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True, help="step_groups.jsonl from the original GiGPO run")
    ap.add_argument("--steps", type=int, default=20, help="use training steps 1..N")
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()

    steps = load(args.diag, args.steps)
    sizes = np.concatenate([np.array(s["sizes"]) for s in steps])
    stds = np.concatenate([np.array(s["stds"], dtype=float) for s in steps])
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
        (out / "diagnostic_report.md").write_text("# Anchor-group diagnostic\n\n" + msg + "\n")
        raise SystemExit(msg)

    min_std = float(np.percentile(stds[spread], 25))
    tau = float(np.median(stds[spread]))

    # sample-weighted shares
    n_samples = sizes.sum()
    signal_samples = sizes[spread].sum()
    touched = spread & ((sizes < MIN_GROUP_SIZE) | (stds < min_std))
    touched_share = sizes[touched].sum() / signal_samples
    zero_share = 1 - signal_samples / n_samples

    low_impact = touched_share < LOW_IMPACT_MARK
    (out / "thresholds.env").write_text(
        f"# from {args.diag}, training steps 1..{args.steps} ({len(steps)} steps)\n"
        f"export MIN_GROUP_SIZE={MIN_GROUP_SIZE}\n"
        f"export MIN_RETURN_STD={min_std:.6g}\n"
        f"export WEIGHT_TAU={tau:.6g}\n")

    size_counts = {k: int((sizes == k).sum()) for k in range(1, int(sizes.max()) + 1) if (sizes == k).any()}
    lines = [
        "# Anchor-group diagnostic",
        "",
        f"Source: `{args.diag}`, training steps 1..{args.steps} ({len(steps)} steps found).",
        "",
        f"- Anchor groups: {len(sizes)}; step samples: {int(n_samples)}",
        f"- Groups of size 1: {np.mean(sizes == 1):.1%}; size 2: {np.mean(sizes == 2):.1%}; size >= 3: {np.mean(sizes >= 3):.1%}",
        f"- Groups with >= 2 members but equal returns: {np.mean(multi & ~spread):.1%}",
        f"- Step samples with zero micro advantage (size 1 or equal returns): {zero_share:.1%}",
        f"- Spread-group return std: p10 {np.percentile(stds[spread], 10):.4g}, p25 {min_std:.4g}, "
        f"median {tau:.4g}, p75 {np.percentile(stds[spread], 75):.4g}, p90 {np.percentile(stds[spread], 90):.4g}",
        "",
        "## Thresholds (predeclared rule)",
        "",
        f"- MIN_GROUP_SIZE = {MIN_GROUP_SIZE}",
        f"- MIN_RETURN_STD = {min_std:.6g} (p25 of spread-group std)",
        f"- WEIGHT_TAU = {tau:.6g} (median of spread-group std)",
        "",
        "## Threshold impact forecast",
        "",
        f"The threshold filter would remove the micro advantage of {touched_share:.1%} of the step samples "
        f"that currently receive one (low-impact marker: {LOW_IMPACT_MARK:.0%}).",
        "",
        ("**Low threshold impact expected**: still train both extensions; the weighted variant can change other groups." if low_impact else
         "**Noticeable threshold impact expected**: train both extensions."),
        "",
        "Group-size counts: " + ", ".join(f"{k}: {v}" for k, v in size_counts.items()),
    ]
    (out / "diagnostic_report.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    ks = list(size_counts)
    axes[0].bar(ks, [size_counts[k] / len(sizes) for k in ks], color=BLUE, width=0.7)
    axes[0].set_xlabel("Anchor-group size", color=MUTED)
    axes[0].set_ylabel("Share of groups", color=MUTED)
    axes[1].hist(stds[spread], bins=40, color=BLUE)
    for v, name in [(min_std, "MIN_RETURN_STD"), (tau, "WEIGHT_TAU")]:
        axes[1].axvline(v, color=INK, linewidth=1, linestyle="--")
        axes[1].annotate(name, (v, axes[1].get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points",
                         fontsize=8, color=INK)
    axes[1].set_xlabel("Return std within group (groups with spread)", color=MUTED)
    axes[1].set_ylabel("Groups", color=MUTED)
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
