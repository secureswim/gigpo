"""Download the authors' public W&B runs (or your own) as history.csv files.

    pip install wandb
    python3 analysis/fetch_wandb.py ENTITY/PROJECT/RUN_ID --name paper_gigpo_seed0
    python3 analysis/fetch_wandb.py ENTITY/PROJECT/RUN_ID --name paper_grpo_seed0

The authors link W&B *reports*, not run paths. To get a run path, open the report
from the upstream README (GiGPO ALFWorld 1.5B:
https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/78zz4sc9),
click a run in a panel's legend, and copy ENTITY/PROJECT/RUN_ID from the run's
URL (wandb.ai/ENTITY/PROJECT/runs/RUN_ID). Public runs need no API key, though
wandb may still ask you to log in once.

Each run is saved as <out>/<name>/history.csv. Name the folder
<condition>_seed<k> (for example ``paper_gigpo_seed0``) so summarize.py and
plot_curves.py can read it next to your own runs, e.g.
``python3 analysis/plot_curves.py --runs reference_runs``.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_path", help="ENTITY/PROJECT/RUN_ID")
    ap.add_argument("--name", required=True, help="output folder name, e.g. paper_gigpo_seed0")
    ap.add_argument("--out", default="reference_runs")
    args = ap.parse_args()

    import pandas as pd
    import wandb

    run = wandb.Api(timeout=60).run(args.run_path)
    rows = list(run.scan_history())
    if not rows:
        raise SystemExit("Run has no history rows")
    df = pd.DataFrame(rows)
    dest = Path(args.out) / args.name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "history.csv", index=False)
    (dest / "source.txt").write_text(f"wandb run: {args.run_path}\nname: {run.name}\nconfig: {dict(run.config)}\n")
    cols = [c for c in df.columns if "success" in c]
    print(f"Saved {len(df)} rows to {dest/'history.csv'}; success columns: {cols}")


if __name__ == "__main__":
    main()
