"""End-to-end launch check without a GPU.

Runs scripts/run_condition.sh -> the upstream example script in the patched
tree -> a stand-in for ``python3 -m verl.trainer.main_ppo`` that composes the
full Hydra config with every override exactly as the real trainer would. This
catches unknown keys, override-parsing errors and wrong values before a GPU
run. (It caught the upstream scripts forwarding the engine name ``vllm`` to
Hydra as an override.)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hydra")
ROOT = Path(__file__).resolve().parents[1]

COMPOSE = r'''
import sys, json
from hydra import compose, initialize_config_dir
cfg_dir, overrides = sys.argv[1], sys.argv[2:]
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    c = compose(config_name="ppo_trainer", overrides=overrides)
a, t, r = c.algorithm, c.trainer, c.actor_rollout_ref
print("COMPOSED " + json.dumps(dict(
    adv=a.adv_estimator, gamma=a.gamma, mode=a.gigpo.mode, w=a.gigpo.step_advantage_w,
    filt=a.gigpo.group_filter, tau=a.gigpo.weight_tau, ms=a.gigpo.min_return_std, mg=a.gigpo.min_group_size,
    diag_dir=a.gigpo.diag_dir, dyn=a.filter_groups.enable, seed=c.env.seed, n=c.env.rollout.n,
    max_steps=c.env.max_steps, epochs=t.total_epochs, save=t.save_freq, test=t.test_freq,
    keep=t.max_actor_ckpt_to_keep, resume=t.resume_mode, gpus=t.n_gpus_per_node,
    tp=r.rollout.tensor_model_parallel_size, engine=r.rollout.name, lr=r.actor.optim.lr,
    val_temp=r.rollout.val_kwargs.temperature, clip_high=r.actor.clip_ratio_high,
    bs=c.data.train_batch_size, logger=list(t.logger))))
'''

SHIM = """#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "examples.data_preprocess.prepare" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "verl.trainer.main_ppo" ]]; then shift 2; exec {py} {compose} "$PWD/verl/trainer/config" "$@"; fi
exec {py} "$@"
"""


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("launch")
    ext = base / "verl-agent-ext"
    r = subprocess.run(["bash", "setup/build_ext_tree.sh"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "EXT_TREE": str(ext)})
    assert r.returncode == 0, r.stdout + r.stderr
    shim = base / "bin"
    shim.mkdir()
    (base / "compose.py").write_text(COMPOSE)
    (shim / "python3").write_text(SHIM.format(py=sys.executable, compose=base / "compose.py"))
    (shim / "python3").chmod(0o755)
    return {"EXT_TREE": str(ext), "RUN_ROOT": str(base / "runs"), "PATH": f"{shim}:{os.environ['PATH']}",
            "WANDB": "0", "MIN_GROUP_SIZE": "3", "MIN_RETURN_STD": "0.25", "WEIGHT_TAU": "0.5", "SEED": "3"}


def launch(condition, env):
    r = subprocess.run(["bash", "scripts/run_condition.sh", condition], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, **env})
    out = r.stdout + r.stderr
    lines = [x for x in out.splitlines() if x.startswith("COMPOSED ")]
    assert r.returncode == 0 and lines, out[-3000:]
    return json.loads(lines[-1][len("COMPOSED "):])


def test_every_condition_composes_with_intended_settings(env):
    c = {k: launch(k, env) for k in ["gigpo", "grpo", "gigpo_threshold", "gigpo_weighted", "gigpo_dynamic"]}
    for k, v in c.items():  # shared settings: paper values + our launcher policy
        assert (v["seed"], v["n"], v["max_steps"], v["epochs"], v["bs"]) == (3, 8, 50, 150, 16), k
        assert (v["save"], v["test"], v["keep"], v["resume"]) == (10, 5, 2, "auto"), k
        assert (v["engine"], v["lr"], v["val_temp"], v["gpus"], v["tp"]) == ("vllm", 1e-6, 0.4, 2, 2), k
        assert v["logger"] == ["console"], k
    for k in ["gigpo", "gigpo_threshold", "gigpo_weighted", "gigpo_dynamic"]:
        assert (c[k]["adv"], c[k]["gamma"], c[k]["mode"], c[k]["w"]) == ("gigpo", 0.95, "mean_std_norm", 1.0)
        assert c[k]["diag_dir"].endswith(f"{k}_seed3/gigpo_diag")
    assert c["grpo"]["adv"] == "grpo"
    assert c["gigpo"]["filt"] == "none" and not c["gigpo"]["dyn"]
    assert (c["gigpo_threshold"]["filt"], c["gigpo_threshold"]["mg"], c["gigpo_threshold"]["ms"]) == ("threshold", 3, 0.25)
    assert (c["gigpo_weighted"]["filt"], c["gigpo_weighted"]["tau"]) == ("weighted", 0.5)
    assert c["gigpo_dynamic"]["dyn"] and c["gigpo_dynamic"]["clip_high"] == 0.28
    assert c["gigpo"]["clip_high"] == 0.2


def test_single_gpu_settings(env):
    v = launch("gigpo", {**env, "N_GPUS": "1", "MICRO_BS": "8"})
    assert (v["gpus"], v["tp"]) == (1, 1)
