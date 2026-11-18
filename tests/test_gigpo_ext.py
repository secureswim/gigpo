"""CPU tests for the anchor-group extension. Run from the project root:

    python3 -m pytest tests -q

Needs only numpy, torch (CPU build is enough) and pytest. ``verl`` is stubbed
because core_gigpo only imports its DataProto type.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream-verl-agent"

if "verl" not in sys.modules:
    stub = types.ModuleType("verl")
    stub.DataProto = object
    sys.modules["verl"] = stub
sys.path.insert(0, str(UPSTREAM))
from gigpo import core_gigpo  # noqa: E402  (pristine upstream)

_spec = importlib.util.spec_from_file_location("gigpo_ext", ROOT / "extension" / "gigpo_ext.py")
gigpo_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gigpo_ext)


# ------------------------------------------------------------------ helpers
def make_batch(groups, L=6, seed=0):
    """groups: list of (episode_id, anchor_obs, [returns...]) -> GiGPO inputs.

    Each listed return becomes one step sample. The episode reward of a sample
    is placed on its last valid token.
    """
    rng = np.random.default_rng(seed)
    anchor, index, step_r = [], [], []
    for ep, obs, rets in groups:
        for r in rets:
            anchor.append(obs)
            index.append(ep)
            step_r.append(r)
    bs = len(step_r)
    lengths = rng.integers(2, L + 1, size=bs)
    eos = torch.zeros(bs, L)
    tlr = torch.zeros(bs, L)
    for i, n in enumerate(lengths):
        eos[i, :n] = 1
        tlr[i, n - 1] = float(rng.choice([0.0, 10.0]))
    return dict(token_level_rewards=tlr, step_rewards=torch.tensor(step_r, dtype=torch.float32),
                eos_mask=eos, anchor_obs=np.array(anchor, dtype=object), index=np.array(index, dtype=object))


def random_batch(seed=1, n_episode_groups=4, n_obs=6, n_samples=80):
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(n_samples):
        ep = f"ep{rng.integers(n_episode_groups)}"
        obs = f"obs{rng.integers(n_obs)}"
        groups.append((ep, obs, [float(rng.choice([0.0, 10 * 0.95 ** rng.integers(0, 10), -0.1]))]))
    return make_batch(groups, seed=seed)


def micro_part(scores, b, mode):
    remove_std = mode == "mean_norm"
    ep = core_gigpo.episode_norm_reward(b["token_level_rewards"], b["eos_mask"], b["index"], 1e-6, remove_std)
    return scores - ep


def per_sample(adv, eos):
    return gigpo_ext._per_sample_scalar(adv, eos)


# ------------------------------------------------------------------ equivalence
@pytest.mark.parametrize("mode", ["mean_std_norm", "mean_norm"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_filter_none_is_identical_to_upstream(mode, seed):
    b = random_batch(seed)
    ref, _ = core_gigpo.compute_gigpo_outcome_advantage(mode=mode, **b)
    ext, _, metrics, record = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode=mode, group_filter="none", **b)
    assert torch.equal(ref, ext)
    assert metrics["gigpo_diag/micro_mass_ratio"] == pytest.approx(1.0)
    assert set(record["weights"]) == {1.0}


# ------------------------------------------------------------------ upstream behaviour the design relies on
def test_degenerate_groups_already_get_zero_micro_advantage():
    b = make_batch([("e", "solo", [5.0]),             # size 1
                    ("e", "flat", [3.0, 3.0, 3.0]),   # all returns equal
                    ("e", "spread", [0.0, 10.0])])
    uids = core_gigpo.build_step_group(b["anchor_obs"], b["index"])
    step = per_sample(core_gigpo.step_norm_reward(b["step_rewards"], b["eos_mask"], uids, 1e-6, False), b["eos_mask"])
    assert np.allclose(step[:4], 0.0)
    assert np.all(np.abs(step[4:]) > 0.5)


def test_std_normalisation_is_scale_free_for_size_two_groups():
    b = make_batch([("e", "tiny", [1.0, 1.01]), ("e", "huge", [0.0, 10.0])])  # gap 0.01 vs 10
    uids = core_gigpo.build_step_group(b["anchor_obs"], b["index"])
    step = per_sample(core_gigpo.step_norm_reward(b["step_rewards"], b["eos_mask"], uids, 1e-6, False), b["eos_mask"])
    assert np.allclose(np.abs(step), 1 / np.sqrt(2), atol=1e-3)


# ------------------------------------------------------------------ threshold condition
def test_threshold_removes_only_low_signal_groups_and_keeps_episode_advantage():
    b = make_batch([("e1", "small", [0.0, 10.0]),
                    ("e1", "lowstd", [9.0, 9.05, 9.1]),
                    ("e1", "keep", [0.0, 5.0, 10.0]),
                    ("e2", "keep", [0.0, 10.0, 10.0])])
    mode = "mean_std_norm"
    ref, *_ = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode=mode, group_filter="none", **b)
    out, _, m, _ = gigpo_ext.compute_gigpo_outcome_advantage_ext(
        mode=mode, group_filter="threshold", min_group_size=3, min_return_std=0.5, **b)
    micro_ref = per_sample(micro_part(ref, b, mode), b["eos_mask"])
    micro_out = per_sample(micro_part(out, b, mode), b["eos_mask"])
    assert np.allclose(micro_out[:5], 0.0)                    # small + lowstd removed
    assert np.allclose(micro_out[5:], micro_ref[5:], atol=1e-6)  # others untouched
    assert m["gigpo_diag/frac_samples_micro_kept"] == pytest.approx(6 / 11)


def test_rule_v2_threshold_keeps_size_two_outcome_pairs_and_drops_penalty_only_groups():
    # Real return values (float32, as in the trainer): win 3 steps later vs never win;
    # never win, one invalid action; two wins one step apart, 10 steps from the goal.
    b = make_batch([("e1", "pair", [10 * 0.95 ** 3, 0.0]),
                    ("e1", "pen", [0.0, -0.1]),
                    ("e1", "pen3", [0.0, 0.0, -0.1]),
                    ("e2", "timing", [10 * 0.95 ** 10, 10 * 0.95 ** 11])])
    mode = "mean_std_norm"
    ref, *_ = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode=mode, group_filter="none", **b)
    out, _, _, rec = gigpo_ext.compute_gigpo_outcome_advantage_ext(
        mode=mode, group_filter="threshold", min_group_size=2, min_return_std=0.075, **b)
    mr = per_sample(micro_part(ref, b, mode), b["eos_mask"])
    mo = per_sample(micro_part(out, b, mode), b["eos_mask"])
    assert np.allclose(mo[:2], mr[:2], atol=1e-6) and np.abs(mo[:2]).min() > 0.5   # outcome pair kept
    assert np.allclose(mo[2:7], 0.0)                                                # penalty-only removed
    assert np.allclose(mo[7:], mr[7:], atol=1e-6)                                   # timing pair kept
    by_size = sorted(zip(rec["sizes"], rec["mins"], rec["maxs"]))
    assert (3, -0.1, 0.0) in by_size and (2, -0.1, 0.0) in by_size


def test_threshold_never_keeps_size_one_groups():
    stats = gigpo_ext.group_statistics(np.array([1.0, 2.0, 3.0]), np.array(["a", "b", "b"], dtype=object))
    w = gigpo_ext.group_weights(stats, "threshold", min_group_size=1, min_return_std=0.0)
    assert w.tolist() == [0.0, 1.0]


# ------------------------------------------------------------------ weighted condition
def test_weighted_preserves_micro_mass_and_orders_by_divergence():
    b = make_batch([("e", "low", [5.0, 5.2, 5.4, 5.6]),
                    ("e", "high", [0.0, 10.0, 0.0, 10.0]),
                    ("e", "solo", [3.0])])
    mode = "mean_std_norm"
    ref, *_ = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode=mode, group_filter="none", **b)
    out, _, m, rec = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode=mode, group_filter="weighted", weight_tau=1.0, **b)
    mr = per_sample(micro_part(ref, b, mode), b["eos_mask"])
    mo = per_sample(micro_part(out, b, mode), b["eos_mask"])
    assert np.abs(mo).sum() == pytest.approx(np.abs(mr).sum(), rel=1e-5)
    assert np.abs(mo[4:8]).mean() > np.abs(mo[:4]).mean()
    assert m["gigpo_diag/micro_mass_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert rec["weights"][2] == 0.0


def test_weighted_requires_tau():
    b = random_batch()
    with pytest.raises(ValueError):
        gigpo_ext.compute_gigpo_outcome_advantage_ext(mode="mean_std_norm", group_filter="weighted", **b)


def test_unknown_filter_rejected():
    with pytest.raises(ValueError):
        gigpo_ext.compute_gigpo_outcome_advantage_ext(mode="mean_std_norm", group_filter="bogus", **random_batch())


# ------------------------------------------------------------------ diagnostics
def test_diagnostic_counts():
    b = make_batch([("e1", "a", [1.0]),
                    ("e1", "b", [2.0, 2.0]),
                    ("e1", "c", [0.0, 1.0, 2.0]),
                    ("e2", "a", [4.0, 6.0])])
    _, _, m, rec = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode="mean_std_norm", **b)
    assert m["gigpo_diag/num_step_groups"] == 4
    assert m["gigpo_diag/num_step_samples"] == 8
    assert m["gigpo_diag/frac_groups_size1"] == pytest.approx(0.25)
    assert m["gigpo_diag/frac_groups_size2"] == pytest.approx(0.5)
    assert m["gigpo_diag/frac_groups_zero_std"] == pytest.approx(0.25)
    assert m["gigpo_diag/frac_groups_zero_signal"] == pytest.approx(0.5)
    assert m["gigpo_diag/frac_samples_zero_signal"] == pytest.approx(3 / 8)
    assert sorted(rec["sizes"]) == [1, 2, 2, 3]
    assert m["gigpo_diag/return_std_p50"] == pytest.approx(np.median([1.0, np.sqrt(2)]))


def test_dump_record(tmp_path):
    _, _, _, rec = gigpo_ext.compute_gigpo_outcome_advantage_ext(mode="mean_std_norm", **random_batch())
    gigpo_ext.dump_record(rec, str(tmp_path), 7)
    gigpo_ext.dump_record(rec, str(tmp_path), 8)
    lines = (tmp_path / "step_groups.jsonl").read_text().splitlines()
    assert [json.loads(x)["step"] for x in lines] == [7, 8]


# ------------------------------------------------------------------ build + scripts
def _bash(cmd, env=None):
    return subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


@pytest.fixture(scope="module")
def ext_tree(tmp_path_factory):
    dest = tmp_path_factory.mktemp("ext") / "verl-agent-ext"
    r = _bash("bash setup/build_ext_tree.sh", {"EXT_TREE": str(dest)})
    assert r.returncode == 0, r.stdout + r.stderr
    return dest


def test_patched_tree_builds_and_contains_hooks(ext_tree):
    trainer = (ext_tree / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert "gigpo_ext.compute_gigpo_outcome_advantage_ext" in trainer
    assert "env/steps_collected" in trainer
    assert "group_filter" in (ext_tree / "verl/trainer/config/ppo_trainer.yaml").read_text()
    assert (ext_tree / "EXT_BUILD_INFO").exists()
    # pristine snapshot untouched
    assert "gigpo_ext" not in (UPSTREAM / "verl/trainer/ppo/ray_trainer.py").read_text()


def test_patched_config_defaults_are_upstream_behaviour(ext_tree):
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.load(ext_tree / "verl/trainer/config/ppo_trainer.yaml")
    assert cfg.algorithm.gigpo.group_filter == "none"
    assert cfg.algorithm.gigpo.mode == "mean_norm"  # upstream default; scripts set mean_std_norm


def _dry(condition, env, ext_tree):
    r = _bash(f"bash scripts/run_condition.sh {condition}",
              {"DRY_RUN": "1", "EXT_TREE": str(ext_tree), **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_conditions_differ_only_in_method_settings(ext_tree):
    env = {"MIN_GROUP_SIZE": "2", "MIN_RETURN_STD": "0.075", "WEIGHT_TAU": "0.5", "SEED": "0"}
    runs = {c: _dry(c, env, ext_tree) for c in ["gigpo", "gigpo_threshold", "gigpo_weighted"]}

    def shared(lines):
        return {x for x in lines[1:] if not x.startswith(("algorithm.", "trainer.experiment_name",
                                                         "trainer.default_local_dir"))}
    base = shared(runs["gigpo"])
    for c, lines in runs.items():
        assert shared(lines) == base, c
    assert "env.seed=0" in base
    assert "trainer.save_freq=5" in base
    assert "algorithm.gigpo.group_filter=threshold" in runs["gigpo_threshold"]
    assert "algorithm.gigpo.min_return_std=0.075" in runs["gigpo_threshold"]
    assert "algorithm.gigpo.weight_tau=0.5" in runs["gigpo_weighted"]
    assert {lines[0] for lines in runs.values()} == {"examples/gigpo_trainer/run_alfworld.sh"}


def test_out_of_scope_launches_are_rejected(ext_tree):
    for cond, seed in [("grpo", "0"), ("gigpo_dynamic", "0"), ("gigpo", "1")]:
        r = _bash(f"bash scripts/run_condition.sh {cond}", {"DRY_RUN": "1", "EXT_TREE": str(ext_tree), "SEED": seed})
        assert r.returncode != 0, (cond, seed)


def test_threshold_condition_requires_thresholds(ext_tree):
    r = _bash("bash scripts/run_condition.sh gigpo_threshold", {"DRY_RUN": "1", "EXT_TREE": str(ext_tree)})
    assert r.returncode != 0
