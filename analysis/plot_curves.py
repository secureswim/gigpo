"""Learning curves: validation success vs. training step and vs. environment steps.

    python3 analysis/plot_curves.py --runs runs --out results/figures [--threshold 0.70]

One line per condition (mean over seeds; band = min..max over seeds when there
is more than one). Also plots training diagnostics (episode length, entropy,
gradient norm) and, for GiGPO conditions, the anchor-group diagnostics.
Colours are fixed per condition so they never change between figures.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import SUCCESS, add_env_steps, condition_and_seed, discover_runs, load_run  # noqa: E402

# Fixed condition -> colour (validated categorical palette, light mode, slots 1-5).
COLORS = {"gigpo": "#2a78d6", "grpo": "#eb6834", "gigpo_threshold": "#1baf7a",
          "gigpo_weighted": "#eda100", "gigpo_dynamic": "#e87ba4"}
LABELS = {"gigpo": "GiGPO", "grpo": "GRPO", "gigpo_threshold": "GiGPO + threshold filter",
          "gigpo_weighted": "GiGPO + divergence weight", "gigpo_dynamic": "GiGPO + dynamic sampling"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def load_all(root):
    data = {}
    for rd in discover_runs(Path(root)):
        cond, seed = condition_and_seed(rd)
        data.setdefault(cond, []).append(add_env_steps(load_run(rd)))
    return data


def plot_metric(data, metric, x, ax, label_last=True):
    for cond, runs in data.items():
        series = []
        for df in runs:
            if metric not in df.columns:
                continue
            s = df[[metric, "cum_env_steps"]].dropna(subset=[metric])
            idx = s.index if x == "step" else s["cum_env_steps"]
            series.append(pd.Series(s[metric].values, index=idx.values))
        if not series:
            continue
        color = COLORS.get(cond, "#6250d6")
        if x == "step":
            frame = pd.concat(series, axis=1)
            mean, lo, hi = frame.mean(axis=1), frame.min(axis=1), frame.max(axis=1)
            ax.plot(mean.index, mean.values, color=color, linewidth=2, label=f"{LABELS.get(cond, cond)} (n={len(series)})")
            if len(series) > 1:
                ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.15, linewidth=0)
            last_x, last_y = mean.index[-1], mean.values[-1]
        else:  # env steps differ per run, so draw each seed
            for i, s in enumerate(series):
                ax.plot(s.index, s.values, color=color, linewidth=2, alpha=0.9 if i == 0 else 0.5,
                        label=f"{LABELS.get(cond, cond)} (n={len(series)})" if i == 0 else None)
            last_x, last_y = series[0].index[-1], series[0].values[-1]
        if label_last and len(data) <= 4:
            ax.annotate(LABELS.get(cond, cond), (last_x, last_y), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    data = load_all(args.runs)
    if not data:
        sys.exit(f"No runs under {args.runs}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    for ax, x, xl in [(axes[0], "step", "Training step"), (axes[1], "env", "Environment steps (cumulative)")]:
        plot_metric(data, SUCCESS, x, ax)
        style(ax, xl, "Validation success rate")
        if args.threshold is not None:
            ax.axhline(args.threshold, color=MUTED, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("ALFWorld validation success, Qwen2.5-1.5B-Instruct", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "success_curves.png")
    plt.close(fig)

    diag = [("episode/length/mean", "Episode length (train)"), ("actor/entropy", "Policy entropy"),
            ("actor/grad_norm", "Gradient norm"), ("episode/reward/mean", "Training episode reward")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    for ax, (m, title) in zip(axes.flat, diag):
        plot_metric(data, m, "step", ax, label_last=False)
        style(ax, "Training step", title)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_diagnostics.png")
    plt.close(fig)

    gd = [("gigpo_diag/frac_samples_zero_signal", "Share of step samples with zero micro signal"),
          ("gigpo_diag/frac_groups_size2", "Share of anchor groups of size 2"),
          ("gigpo_diag/mean_group_size", "Mean anchor-group size"),
          ("gigpo_diag/mean_abs_step_over_episode", "|micro| / |macro| advantage")]
    gdata = {c: r for c, r in data.items() if c != "grpo"}
    if any(m in df.columns for runs in gdata.values() for df in runs for m, _ in gd):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
        for ax, (m, title) in zip(axes.flat, gd):
            plot_metric(gdata, m, "step", ax, label_last=False)
            style(ax, "Training step", title)
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "anchor_group_diagnostics.png")
        plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
