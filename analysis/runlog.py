"""Read training runs into tables and compute the project's outcome measures.

Sources
  * ``runs/<condition>_seed<k>/train.log``  console log written by run_condition.sh
  * a CSV exported by ``analysis/fetch_wandb.py`` (same column names, full precision)

Console lines look like ``step:5 - val/success_rate:0.312 - actor/grad_norm:0.871 ...``
(values rounded to 3 decimals by verl's console logger). Ray may prefix lines with
``(main_task pid=...)``. The pre-training validation is printed once as
``Initial validation metrics: {...}`` and is stored as step 0.

When a run was resumed, a step can appear twice; the last occurrence wins.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

SUCCESS = "val/success_rate"
ENV_STEPS = "env/steps_collected"
STEP_RE = re.compile(r"step:(\d+) - (.*)$")
INIT_RE = re.compile(r"Initial validation metrics: (\{.*\})")
KV_RE = re.compile(r"'([^']+)':\s*(?:np\.float\d*\()?(-?[\d.]+(?:e[-+]?\d+)?|nan)")


def parse_console_log(path: Path) -> pd.DataFrame:
    rows: Dict[int, Dict[str, float]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = INIT_RE.search(line)
            if m:
                rows[0] = {k: float(v) for k, v in KV_RE.findall(m.group(1))}
                continue
            m = STEP_RE.search(line.rstrip("\n"))
            if not m:
                continue
            step = int(m.group(1))
            rec = {}
            for part in m.group(2).split(" - "):
                if ":" not in part:
                    continue
                k, v = part.rsplit(":", 1)
                try:
                    rec[k.strip()] = float(v)
                except ValueError:
                    pass
            if rec:
                rows[step] = rec  # last occurrence wins (resumed runs)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "step"
    return df


def load_run(path: Path) -> pd.DataFrame:
    """Accept a run directory, a train.log, or a W&B-exported CSV."""
    path = Path(path)
    if path.is_dir():
        csv = path / "history.csv"
        path = csv if csv.exists() else path / "train.log"
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        step_col = "_step" if "_step" in df.columns else "step"
        df = df.groupby(step_col).last()
        df.index.name = "step"
        return df.sort_index()
    return parse_console_log(path)


def add_env_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative environment steps at the end of each training step."""
    df = df.copy()
    if ENV_STEPS in df.columns:
        per = df[ENV_STEPS].fillna(0)
        per.loc[per.index == 0] = 0
        df["cum_env_steps"] = per.cumsum()
    else:
        df["cum_env_steps"] = math.nan
    return df


def condition_and_seed(run_dir: Path):
    m = re.match(r"(.+)_seed(\d+)", Path(run_dir).name)
    return (m.group(1), int(m.group(2))) if m else (Path(run_dir).name, None)


def summarize_run(df: pd.DataFrame, threshold: Optional[float], tail_points: int = 3) -> Dict[str, float]:
    df = add_env_steps(df)
    out: Dict[str, float] = {}
    if SUCCESS in df.columns:
        val = df[SUCCESS].dropna()
        out["val_points"] = len(val)
        out["final_success"] = float(val.iloc[-1]) if len(val) else math.nan
        out[f"mean_last{tail_points}_success"] = float(val.iloc[-tail_points:].mean()) if len(val) else math.nan
        out["best_success"] = float(val.max()) if len(val) else math.nan
        out["best_step"] = float(val.idxmax()) if len(val) else math.nan
        if threshold is not None:
            hit = val[val >= threshold]
            out["steps_to_threshold"] = float(hit.index[0]) if len(hit) else math.nan
            out["env_steps_to_threshold"] = (float(df.loc[hit.index[0], "cum_env_steps"])
                                             if len(hit) else math.nan)
    train = df[df.index > 0]
    out["train_steps"] = float(train.index.max()) if len(train) else 0.0
    out["total_env_steps"] = float(df["cum_env_steps"].max()) if df["cum_env_steps"].notna().any() else math.nan
    for col, name in [("episode/length/mean", "episode_length"), ("actor/entropy", "entropy"),
                      ("actor/grad_norm", "grad_norm"), ("episode/reward/mean", "train_reward")]:
        if col in train.columns and train[col].notna().any():
            s = train[col].dropna()
            third = max(len(s) // 3, 1)
            out[f"{name}_first_third"] = float(s.iloc[:third].mean())
            out[f"{name}_last_third"] = float(s.iloc[-third:].mean())
            out[f"{name}_std"] = float(s.std()) if len(s) > 1 else math.nan
    if "actor/grad_norm" in train.columns and train["actor/grad_norm"].notna().any():
        out["grad_norm_max"] = float(train["actor/grad_norm"].max())
    return out


APPROVED_RUN_NAMES = {"gigpo_seed0", "gigpo_threshold_seed0", "gigpo_weighted_seed0"}


def discover_runs(root: Path) -> List[Path]:
    """Only the three full runs in this handoff; exclude smoke and old runs."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and
                  p.name in APPROVED_RUN_NAMES and
                  ((p / "train.log").exists() or (p / "history.csv").exists()))


def summary_table(run_dirs: Iterable[Path], threshold: Optional[float]) -> pd.DataFrame:
    recs = []
    for rd in run_dirs:
        cond, seed = condition_and_seed(rd)
        rec = {"run": Path(rd).name, "condition": cond, "seed": seed}
        rec.update(summarize_run(load_run(rd), threshold))
        recs.append(rec)
    return pd.DataFrame(recs)
