"""Integration test of the patched trainer's advantage path with a real DataProto
and the real composed config. Runs in a subprocess against the patched tree.

Skipped automatically when verl's runtime dependencies (ray, tensordict,
transformers, torchdata, codetiming) are not installed, e.g. on a laptop. On the
GPU machine they are installed, so preflight runs it.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import sys
tree = sys.argv[1]
sys.path.insert(0, tree)
import numpy as np, torch
from omegaconf import OmegaConf
from verl import DataProto
import verl.trainer.ppo.ray_trainer as rt
from gigpo import gigpo_ext

assert "verl-agent-ext" in rt.__file__ or tree in rt.__file__, rt.__file__
rng = np.random.default_rng(0)
bs, L = 96, 8
uid = np.array([f"e{i % 4}" for i in range(bs)], dtype=object)
anchor = np.array([f"o{rng.integers(5)}" for _ in range(bs)], dtype=object)
mask = torch.ones(bs, L)
mask[:, 6:] = 0
tlr = torch.zeros(bs, L)
tlr[:, 5] = torch.tensor(rng.choice([0.0, 10.0], size=bs), dtype=torch.float32)
step = torch.tensor(rng.choice([0.0, 0.5, 9.5, 10.0], size=bs), dtype=torch.float32)

def batch():
    return DataProto.from_dict(
        tensors={"token_level_rewards": tlr.clone(), "step_rewards": step.clone(), "response_mask": mask.clone(),
                 "responses": torch.zeros(bs, L, dtype=torch.long), "attention_mask": torch.ones(bs, 2 * L)},
        non_tensors={"uid": uid, "anchor_obs": anchor})

cfg = OmegaConf.load(f"{tree}/verl/trainer/config/ppo_trainer.yaml").algorithm.gigpo
GiGPO = rt.AdvantageEstimator.GiGPO

up = rt.compute_advantage(batch(), GiGPO, step_advantage_w=1.0, gigpo_mode="mean_std_norm")
ext = rt.compute_advantage(batch(), GiGPO, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_ext_cfg=cfg)
assert torch.equal(up.batch["advantages"], ext.batch["advantages"]), "filter=none differs from upstream"
m = ext.meta_info["gigpo_diag_metrics"]
assert m["gigpo_diag/num_step_samples"] == bs and m["gigpo_diag/micro_mass_ratio"] == 1.0
assert "sizes" in ext.meta_info["gigpo_diag_record"]

cfg_t = OmegaConf.merge(cfg, {"group_filter": "threshold", "min_group_size": 3, "min_return_std": 1.0})
thr = rt.compute_advantage(batch(), GiGPO, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_ext_cfg=cfg_t)
assert not torch.equal(up.batch["advantages"], thr.batch["advantages"])
assert thr.meta_info["gigpo_diag_metrics"]["gigpo_diag/frac_samples_micro_kept"] < 1.0

cfg_w = OmegaConf.merge(cfg, {"group_filter": "weighted", "weight_tau": 1.0})
wt = rt.compute_advantage(batch(), GiGPO, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_ext_cfg=cfg_w)
assert abs(wt.meta_info["gigpo_diag_metrics"]["gigpo_diag/micro_mass_ratio"] - 1.0) < 1e-6

grpo = rt.compute_advantage(batch(), rt.AdvantageEstimator.GRPO, gigpo_ext_cfg=cfg)
assert "gigpo_diag_metrics" not in grpo.meta_info
print("INTEGRATION OK")
'''


def test_patched_compute_advantage(tmp_path):
    for mod in ["ray", "tensordict", "transformers", "torchdata", "codetiming"]:
        pytest.importorskip(mod)
    tree = tmp_path / "verl-agent-ext"
    r = subprocess.run(["bash", "setup/build_ext_tree.sh"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "EXT_TREE": str(tree)})
    assert r.returncode == 0, r.stdout + r.stderr
    script = tmp_path / "integration.py"
    script.write_text(SCRIPT)
    r = subprocess.run([sys.executable, "-I", str(script), str(tree)], cwd=tmp_path, capture_output=True, text=True)
    assert "INTEGRATION OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
