"""Verify a smoke run: updates happened, validation ran, diagnostics and the
environment-step counter were logged, a checkpoint was written, and the second
launch resumed instead of restarting.

    python3 week1/check_smoke.py runs/gigpo_seed0_smoke
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from runlog import load_run  # noqa: E402


def estimate_hours(df, total_steps=150, test_freq=5):
    """Smoke steps validate every step; a real run validates every 5th step.
    Per real step = (step time - validation time) + validation time / test_freq."""
    cols = {"timing_s/step", "timing_s/testing"}
    if not cols <= set(df.columns):
        return None
    rows = df.loc[df.index > 0, list(cols)].dropna()
    if rows.empty:
        return None
    train = (rows["timing_s/step"] - rows["timing_s/testing"]).mean()
    val = rows["timing_s/testing"].mean()
    return total_steps * (train + val / test_freq) / 3600


def main(run_dir: Path) -> int:
    problems, notes = [], []
    log = run_dir / "train.log"
    if not log.exists():
        print(f"FAIL: {log} missing")
        return 1
    text = log.read_text(errors="replace")
    df = load_run(run_dir)

    def need(cond, msg):
        (notes if cond else problems).append(("ok    " if cond else "FAIL  ") + msg)

    need(0 in df.index and "val/success_rate" in df.columns and df["val/success_rate"].notna().any(),
         "validation success rate logged (metric val/success_rate)")
    need(1 in df.index and 2 in df.index, "training steps 1 and 2 logged")
    for col in ["actor/pg_loss", "actor/grad_norm", "actor/entropy", "env/steps_collected",
                "gigpo_diag/num_step_groups", "gigpo_diag/frac_samples_zero_signal"]:
        need(col in df.columns and df[col].notna().any(), f"metric {col} logged")
    need("Setting global step to 1" in text, "second launch resumed from the step-1 checkpoint")
    ckpts = sorted((run_dir / "checkpoints").glob("global_step_*"))
    need(bool(ckpts), f"checkpoint folder written ({[c.name for c in ckpts]})")
    diag = run_dir / "gigpo_diag" / "step_groups.jsonl"
    if diag.exists():
        steps = [json.loads(x)["step"] for x in diag.read_text().splitlines() if x.strip()]
        need(1 in steps and 2 in steps, f"anchor-group dump has steps {steps}")
    else:
        need(False, f"{diag} written")
    if "Traceback" in text:
        problems.append("FAIL  a Python traceback appears in train.log")

    for line in notes + problems:
        print(line)
    if "val/success_rate" in df.columns:
        print("\nval/success_rate by step:", df["val/success_rate"].dropna().to_dict())
    if "env/steps_collected" in df.columns:
        print("env steps per training step:", df["env/steps_collected"].dropna().to_dict())
    if "timing_s/step" in df.columns:
        print("seconds per training step (incl. validation):", df["timing_s/step"].dropna().to_dict())
        est = estimate_hours(df)
        if est is not None:
            print(f"ESTIMATED_HOURS_PER_RUN={est:.1f}  (150 steps, validation every 5 steps; "
                  "excludes queue time and resume overhead)")
    print("\nSMOKE PASSED" if not problems else "\nSMOKE FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
