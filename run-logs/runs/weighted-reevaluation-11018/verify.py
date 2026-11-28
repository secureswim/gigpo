from pathlib import Path
import json
import math
import re
import sys

root = Path(__file__).resolve().parent
seed = int(sys.argv[1])
current = root / f"weighted-{seed}"
prior = root.parent / "paired-validation-10605"

def read_episodes(directory):
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 128 and {row["slot"] for row in rows} == set(range(128))
    assert all(row["task"] and "/valid_seen/" in row["task"] and row["won"] in (0, 1) for row in rows)
    return sorted(rows, key=lambda row: row["slot"])

def task_ids(rows):
    return [row["task"].split("/valid_seen/", 1)[1] for row in rows]

weighted = read_episodes(current)
baseline = read_episodes(prior / f"baseline-{seed}")
threshold = read_episodes(prior / f"threshold-{seed}")
assert task_ids(weighted) == task_ids(baseline) == task_ids(threshold), "Evaluation task order differs"
log = (current / "eval.log").read_text(errors="replace")
assert "Setting global step to 150" in log and "gigpo_weighted_seed0/checkpoints/global_step_150" in log
scores = re.findall(r"val/success_rate:([0-9.]+)", log)
wins = sum(int(row["won"]) for row in weighted)
assert scores and math.isclose(wins / 128, float(scores[-1]), abs_tol=0.00051)
result = {"evaluation_seed": seed, "environment_seed_base": 1000 + 128 * seed,
          "checkpoint_step": 150, "training_seed": 0, "split": "valid_seen",
          "episodes": 128, "unique_games": len(set(task_ids(weighted))),
          "baseline_wins": sum(int(row["won"]) for row in baseline),
          "threshold_wins": sum(int(row["won"]) for row in threshold),
          "weighted_wins": wins, "tasks_match_original_at_each_slot": True}
(current / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
