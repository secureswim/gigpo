"""Outcome table for all runs, per run and per condition (mean +- std over seeds).

    python3 analysis/summarize.py --runs runs --threshold 0.70 [--out results/summary.csv]

``--threshold`` is the success rate used for "steps to target". Fix it in
knowledge_base/experiment_protocol.md before looking at extension results; it is
not chosen from those results. A run that never reaches it reports NaN, never an
invented value.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import discover_runs, summary_table  # noqa: E402

KEY_COLUMNS = ["final_success", "mean_last3_success", "best_success", "steps_to_threshold",
               "env_steps_to_threshold", "total_env_steps", "episode_length_last_third",
               "entropy_last_third", "grad_norm_std", "train_reward_std"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="directory holding <condition>_seed<k>/ run folders")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", default=None, help="optional CSV path for the per-run table")
    args = ap.parse_args()

    runs = discover_runs(Path(args.runs))
    if not runs:
        sys.exit(f"No runs with train.log or history.csv under {args.runs}")
    df = summary_table(runs, args.threshold)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    cols = [c for c in ["run", "condition", "seed"] + KEY_COLUMNS if c in df.columns]
    print("Per run\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    num = [c for c in KEY_COLUMNS if c in df.columns]
    agg = df.groupby("condition")[num].agg(["mean", "std", "count"])
    print("\nPer condition (over seeds)\n" + agg.to_string(float_format=lambda x: f"{x:.4g}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
