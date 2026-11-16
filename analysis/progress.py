"""Where is each run? Reads runs/<condition>_seed<k>[_tag]/ folders.

    python3 analysis/progress.py                 # table of all runs under runs/
    python3 analysis/progress.py --runs runs
    python3 analysis/progress.py --done runs/gigpo_seed0 150   # exit 0 if finished, 1 if not

Columns: last training step, last validation success and its step, latest
checkpoint, environment steps so far, number of launches, Slurm job of the last
launch, and the last error signature found in train.log (if any).
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import SUCCESS, add_env_steps, load_run  # noqa: E402

ERRORS = ["CUDA out of memory", "OutOfMemoryError", "oom-kill", "Killed", "Traceback",
          "DUE TO TIME LIMIT", "RayTaskError", "Error parsing override"]


def last_step(run_dir: Path) -> int:
    log = run_dir / "train.log"
    if not log.exists():
        return 0
    steps = [int(m) for m in re.findall(r"step:(\d+) - ", log.read_text(errors="replace"))]
    return max(steps, default=0)


def summary(run_dir: Path) -> dict:
    out = {"run": run_dir.name, "last_step": last_step(run_dir)}
    try:
        df = add_env_steps(load_run(run_dir))
        if SUCCESS in df.columns and df[SUCCESS].notna().any():
            val = df[SUCCESS].dropna()
            out["last_val"] = f"{val.iloc[-1]:.3f}@{val.index[-1]}"
        if df["cum_env_steps"].notna().any():
            out["env_steps"] = int(df["cum_env_steps"].max())
    except Exception as exc:  # unreadable or empty log
        out["last_val"] = f"unreadable ({type(exc).__name__})"
    ckpts = sorted((run_dir / "checkpoints").glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    out["checkpoint"] = ckpts[-1].name if ckpts else "-"
    launches = run_dir / "launches.txt"
    if launches.exists():
        heads = [x for x in launches.read_text().splitlines() if x.startswith("=== launch")]
        out["launches"] = len(heads)
        m = re.search(r"slurm_job=(\S+)", heads[-1]) if heads else None
        out["last_job"] = m.group(1) if m else "-"
    log = run_dir / "train.log"
    if log.exists():
        tail = log.read_text(errors="replace")[-20000:]
        hits = [e for e in ERRORS if e in tail]
        out["error"] = hits[-1] if hits else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--done", nargs=2, metavar=("RUN_DIR", "TOTAL_STEPS"))
    args = ap.parse_args()
    if args.done:
        run_dir, total = Path(args.done[0]), int(args.done[1])
        sys.exit(0 if last_step(run_dir) >= total else 1)
    root = Path(args.runs)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "launches.txt").exists()) if root.exists() else []
    if not runs:
        print(f"No runs under {root}")
        return
    cols = ["run", "last_step", "last_val", "checkpoint", "env_steps", "launches", "last_job", "error"]
    rows = [summary(r) for r in runs]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
