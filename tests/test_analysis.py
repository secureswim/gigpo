"""Analysis pipeline tests on synthetic logs (no GPU, no training)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import runlog  # noqa: E402


def write_log(path, success_by_step, env_steps=1000, resumed=False):
    lines = ["Using LocalLogger is deprecated",
             "(main_task pid=1) Initial validation metrics: {'val/test_score/alfworld/AlfredTWEnv': "
             f"np.float64(0.5), 'val/success_rate': np.float64({success_by_step.get(0, 0.0)})}}"]
    for step in range(1, max(success_by_step) + 1):
        parts = [f"step:{step}", f"env/steps_collected:{env_steps:.3f}", f"actor/grad_norm:{0.5 + step / 100:.3f}",
                 f"actor/entropy:{1.0 - step / 100:.3f}", f"episode/length/mean:{30 - step / 10:.3f}",
                 "episode/reward/mean:3.000"]
        if step in success_by_step:
            parts.append(f"val/success_rate:{success_by_step[step]:.3f}")
        lines.append("(main_task pid=1) " + " - ".join(parts))
    if resumed:  # a resumed run re-logs step 2 with a different value; last wins
        lines.append(f"step:2 - env/steps_collected:{env_steps:.3f} - val/success_rate:0.999")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_parse_console_log(tmp_path):
    p = tmp_path / "gigpo_seed0" / "train.log"
    write_log(p, {0: 0.1, 5: 0.4, 10: 0.7})
    df = runlog.load_run(p.parent)
    assert df.loc[0, "val/success_rate"] == pytest.approx(0.1)
    assert df.loc[10, "val/success_rate"] == pytest.approx(0.7)
    assert df.loc[3, "actor/grad_norm"] == pytest.approx(0.53)
    assert np.isnan(df.loc[3, "val/success_rate"])


def test_resumed_step_last_wins(tmp_path):
    p = tmp_path / "r_seed0" / "train.log"
    write_log(p, {2: 0.2, 4: 0.3}, resumed=True)
    assert runlog.load_run(p.parent).loc[2, "val/success_rate"] == pytest.approx(0.999)


def test_summary_steps_and_env_steps_to_threshold(tmp_path):
    p = tmp_path / "gigpo_seed0" / "train.log"
    write_log(p, {0: 0.1, 5: 0.4, 10: 0.72, 15: 0.71}, env_steps=500)
    s = runlog.summarize_run(runlog.load_run(p.parent), threshold=0.7)
    assert s["steps_to_threshold"] == 10
    assert s["env_steps_to_threshold"] == 5000
    assert s["final_success"] == pytest.approx(0.71)
    assert s["best_success"] == pytest.approx(0.72)
    assert s["total_env_steps"] == 7500


def test_never_reaching_threshold_is_nan(tmp_path):
    p = tmp_path / "grpo_seed0" / "train.log"
    write_log(p, {5: 0.2, 10: 0.3})
    s = runlog.summarize_run(runlog.load_run(p.parent), threshold=0.9)
    assert np.isnan(s["steps_to_threshold"]) and np.isnan(s["env_steps_to_threshold"])


def test_wandb_csv_input(tmp_path):
    d = tmp_path / "paper_gigpo_seed0"
    d.mkdir()
    (d / "history.csv").write_text("_step,val/success_rate,env/steps_collected\n0,0.1,\n5,0.5,100\n10,0.8,100\n")
    df = runlog.load_run(d)
    assert df.loc[10, "val/success_rate"] == pytest.approx(0.8)


def test_cli_summarize_and_plot(tmp_path):
    for cond, curve in [("gigpo", {5: 0.5, 10: 0.8}),
                        ("gigpo_threshold", {5: 0.45, 10: 0.75}),
                        ("gigpo_weighted", {5: 0.55, 10: 0.82})]:
        write_log(tmp_path / "runs" / f"{cond}_seed0" / "train.log", curve)
    write_log(tmp_path / "runs" / "gigpo_seed0_smoke" / "train.log", {1: 0.99, 2: 0.99})
    write_log(tmp_path / "runs" / "grpo_seed0" / "train.log", {5: 0.3, 10: 0.6})
    r = subprocess.run([sys.executable, str(ROOT / "analysis/summarize.py"), "--runs", str(tmp_path / "runs"),
                        "--threshold", "0.7", "--out", str(tmp_path / "s.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "gigpo" in r.stdout and (tmp_path / "s.csv").exists()
    summary = (tmp_path / "s.csv").read_text()
    assert "gigpo_seed0_smoke" not in summary and "grpo_seed0" not in summary
    assert "gigpo_threshold_seed0" in summary and "gigpo_weighted_seed0" in summary
    r = subprocess.run([sys.executable, str(ROOT / "analysis/plot_curves.py"), "--runs", str(tmp_path / "runs"),
                        "--out", str(tmp_path / "fig"), "--threshold", "0.7"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig" / "success_curves.png").exists()


def test_choose_thresholds(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "step_groups.jsonl"
    with open(path, "w") as f:
        for step in range(0, 30):
            sizes = rng.integers(1, 6, size=200).tolist()
            stds = [0.0 if s == 1 else float(rng.choice([0.0, rng.exponential(1.0)])) for s in sizes]
            f.write(json.dumps({"step": step, "filter": "none", "sizes": sizes, "stds": stds,
                                "ranges": stds, "means": stds, "weights": [1.0] * 200}) + "\n")
    out = tmp_path / "diag"
    r = subprocess.run([sys.executable, str(ROOT / "analysis/choose_thresholds.py"), "--diag", str(path),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    env = (out / "thresholds.env").read_text()
    assert "MIN_GROUP_SIZE=2" in env and "MIN_RETURN_STD=0.075" in env and "WEIGHT_TAU=" in env
    assert "rule v2" in env and "training steps 1..20 (20 steps)" in env
    assert (out / "anchor_groups.png").exists()
    assert "only the std split is shown" in (out / "diagnostic_report.md").read_text()  # no min/max in record


def _alfworld_like_groups(rng, n_groups=300):
    """Synthetic groups built from the real return structure: 10*0.95^k on a later
    win, 0 otherwise, minus 0.1 when the step's own action was invalid."""
    rows = []
    for _ in range(n_groups):
        n = int(rng.integers(2, 6))
        kind = str(rng.choice(["mixed", "timing", "penalty_fail", "equal"]))
        k = int(rng.integers(0, 45))
        if kind == "mixed":
            r = [10 * 0.95 ** k] + [0.0] * (n - 1)
        elif kind == "timing":
            r = [10 * 0.95 ** (k + int(j)) for j in rng.integers(0, 3, size=n)]
            r[0], r[1] = 10 * 0.95 ** k, 10 * 0.95 ** (k + 1)
        elif kind == "penalty_fail":
            r = [0.0] * (n - 1) + [-0.1]
        else:
            r = [0.0] * n
        rows.append((kind, np.asarray(r, dtype=np.float32).astype(np.float64)))
    return rows


def test_bands_and_forecast_on_alfworld_like_returns(tmp_path):
    rng = np.random.default_rng(3)
    groups = _alfworld_like_groups(rng)
    rec = {"step": 1, "filter": "none", "sizes": [], "stds": [], "ranges": [], "means": [], "mins": [],
           "maxs": [], "weights": []}
    for _, r in groups:
        rec["sizes"].append(len(r))
        rec["stds"].append(float(r.std(ddof=1)))
        rec["ranges"].append(float(np.ptp(r)))
        rec["means"].append(float(r.mean()))
        rec["mins"].append(float(r.min()))
        rec["maxs"].append(float(r.max()))
        rec["weights"].append(1.0)
    path = tmp_path / "step_groups.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "diag"
    r = subprocess.run([sys.executable, str(ROOT / "analysis/choose_thresholds.py"), "--diag", str(path),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    report = (out / "diagnostic_report.md").read_text()
    assert "Mixed outcome" in report and "Penalty-scale spread" in report
    assert "Other (std" not in report          # no unexpected band on clean data
    spread = [(k, x) for k, x in groups if x.std(ddof=1) > 1e-6]
    removed = sum(len(x) for k, x in spread if x.std(ddof=1) < 0.075)
    assert all(k in ("penalty_fail", "timing") for k, x in spread if x.std(ddof=1) < 0.075)
    share = removed / sum(len(x) for _, x in spread)
    assert f"would remove the micro advantage of {share:.1%}" in report
    assert "retired rule v1" in report


def test_rule_constants_match_upstream():
    """Rule v2 hard-codes the reward structure; fail loudly if the pinned settings say otherwise."""
    import choose_thresholds as ct
    script = (ROOT / "upstream-verl-agent/examples/gigpo_trainer/run_alfworld.sh").read_text()
    assert f"invalid_action_penalty_coef={ct.INVALID_ACTION_PENALTY}" in script
    assert f"algorithm.gamma={ct.GAMMA}" in script
    assert f"env.max_steps={ct.MAX_STEPS}" in script
    envs = (ROOT / "upstream-verl-agent/agent_system/environments/env_package/alfworld/envs.py").read_text()
    assert "reward = 10.0 * float(info['won'])" in envs and ct.WIN_REWARD == 10.0
    # penalty-only groups always fall below the threshold; real wins always above the split
    assert ct.PENALTY_ONLY_STD < ct.MIN_RETURN_STD < 2 * ct.PENALTY_ONLY_STD
    min_win_return = ct.WIN_REWARD * ct.GAMMA ** (ct.MAX_STEPS - 1) - ct.INVALID_ACTION_PENALTY
    assert 0.0 < ct.SUCCESS_SPLIT < min_win_return


def test_progress_done_and_table(tmp_path):
    run = tmp_path / "runs" / "gigpo_seed0"
    write_log(run / "train.log", {5: 0.4, 10: 0.6})
    (run / "launches.txt").write_text("=== launch 2026-11-16T00:00:00Z slurm_job=123 node=n1\n")
    (run / "checkpoints" / "global_step_10").mkdir(parents=True)
    prog = [sys.executable, str(ROOT / "analysis/progress.py")]
    assert subprocess.run(prog + ["--done", str(run), "10"]).returncode == 0
    assert subprocess.run(prog + ["--done", str(run), "150"]).returncode == 1
    assert subprocess.run(prog + ["--done", str(tmp_path / "missing"), "1"]).returncode == 1
    r = subprocess.run(prog + ["--runs", str(tmp_path / "runs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "gigpo_seed0" in r.stdout and "global_step_10" in r.stdout and "123" in r.stdout


def test_smoke_budget_estimate():
    import pandas as pd
    sys.path.insert(0, str(ROOT / "week1"))
    import check_smoke
    df = pd.DataFrame({"timing_s/step": [700.0, 720.0], "timing_s/testing": [300.0, 300.0]}, index=[1, 2])
    # train 410 s + 300/5 s = 470 s per step -> 150 steps = 19.58 h
    assert check_smoke.estimate_hours(df) == pytest.approx(150 * 470 / 3600)


def test_no_spread_removes_stale_thresholds(tmp_path):
    path = tmp_path / "step_groups.jsonl"
    path.write_text(json.dumps({"step": 1, "sizes": [1, 2], "stds": [0.0, 0.0]}) + "\n")
    out = tmp_path / "diag"
    out.mkdir()
    (out / "thresholds.env").write_text("export WEIGHT_TAU=1\n")
    r = subprocess.run([sys.executable, str(ROOT / "analysis/choose_thresholds.py"),
                        "--diag", str(path), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode != 0
    assert not (out / "thresholds.env").exists()
    report = (out / "diagnostic_report.md").read_text()
    assert report.startswith("# Anchor-group diagnostic\n\n")
    assert "No anchor group with nonzero return spread" in report
