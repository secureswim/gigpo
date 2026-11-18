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
    bs=c.data.train_batch_size, logger=list(t.logger), val_bs=c.data.val_batch_size,
    mini=r.actor.ppo_mini_batch_size, micro=r.actor.ppo_micro_batch_size_per_gpu,
    ref_offload=r.ref.fsdp_config.param_offload, mem=r.rollout.gpu_memory_utilization)))
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
            "WANDB": "0", "MIN_GROUP_SIZE": "2", "MIN_RETURN_STD": "0.075", "WEIGHT_TAU": "0.5", "SEED": "0"}


def launch(condition, env):
    r = subprocess.run(["bash", "scripts/run_condition.sh", condition], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, **env})
    out = r.stdout + r.stderr
    lines = [x for x in out.splitlines() if x.startswith("COMPOSED ")]
    assert r.returncode == 0 and lines, out[-3000:]
    return json.loads(lines[-1][len("COMPOSED "):])


def test_every_condition_composes_with_intended_settings(env):
    c = {k: launch(k, env) for k in ["gigpo", "gigpo_threshold", "gigpo_weighted"]}
    for k, v in c.items():  # shared settings: paper values + our launcher policy
        assert (v["seed"], v["n"], v["max_steps"], v["epochs"], v["bs"]) == (0, 8, 50, 150, 16), k
        assert (v["save"], v["test"], v["keep"], v["resume"]) == (5, 5, 2, "auto"), k
        assert (v["engine"], v["lr"], v["val_temp"], v["gpus"], v["tp"]) == ("vllm", 1e-6, 0.4, 1, 1), k
        assert (v["val_bs"], v["mini"], v["micro"], v["ref_offload"], v["mem"]) == (128, 256, 16, True, 0.5), k
        assert v["logger"] == ["console"], k
    for k in ["gigpo", "gigpo_threshold", "gigpo_weighted"]:
        assert (c[k]["adv"], c[k]["gamma"], c[k]["mode"], c[k]["w"]) == ("gigpo", 0.95, "mean_std_norm", 1.0)
        assert c[k]["diag_dir"].endswith(f"{k}_seed0/gigpo_diag")
    assert c["gigpo"]["filt"] == "none" and not c["gigpo"]["dyn"]
    assert (c["gigpo_threshold"]["filt"], c["gigpo_threshold"]["mg"], c["gigpo_threshold"]["ms"]) == ("threshold", 2, 0.075)
    assert (c["gigpo_weighted"]["filt"], c["gigpo_weighted"]["tau"]) == ("weighted", 0.5)
    assert c["gigpo"]["clip_high"] == 0.2


def test_hardware_knobs(env):
    v = launch("gigpo", {**env, "N_GPUS": "2", "MICRO_BS": "8", "VAL_BATCH": "32", "REF_OFFLOAD": "False",
                         "GPU_MEM_UTIL": "0.4"})
    assert (v["gpus"], v["tp"], v["micro"], v["val_bs"], v["ref_offload"], v["mem"]) == (2, 2, 8, 32, False, 0.4)
    assert (v["bs"], v["mini"], v["n"]) == (16, 256, 8)  # experiment settings unchanged


@pytest.mark.parametrize("bad", ["data.train_batch_size=4", "actor_rollout_ref.actor.ppo_mini_batch_size=32",
                                 "algorithm.gamma=0.9", "+env.rollout.n=4", "env.seed=1", "trainer.total_epochs=1"])
def test_protected_settings_are_refused(env, bad):
    r = subprocess.run(["bash", "scripts/run_condition.sh", "gigpo", bad], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, **env})
    assert r.returncode == 2 and "protected" in r.stderr


def test_data_preparation_skipped_when_parquet_exists(env, tmp_path):
    home = tmp_path / "home"
    (home / "data/verl-agent/text").mkdir(parents=True)
    for f in ("train.parquet", "test.parquet"):
        (home / "data/verl-agent/text" / f).write_text("x")
    script = Path(env["EXT_TREE"]) / "examples/gigpo_trainer/run_alfworld.sh"
    text = script.read_text()
    assert "if [[ ! -f $HOME/data/verl-agent/text/train.parquet" in text
    # run only the guard: with both files present the prepare command must not run
    guard = text[text.index("if [[ ! -f $HOME"):text.index("fi\n", text.index("if [[ ! -f $HOME")) + 3]
    r = subprocess.run(["bash", "-c", guard.replace("python3 -m", "echo PREPARE_RAN")],
                       capture_output=True, text=True, env={**os.environ, "HOME": str(home)})
    assert "PREPARE_RAN" not in r.stdout
    r = subprocess.run(["bash", "-c", guard.replace("python3 -m", "echo PREPARE_RAN")],
                       capture_output=True, text=True, env={**os.environ, "HOME": str(tmp_path / "empty")})
    assert "PREPARE_RAN" in r.stdout


@pytest.mark.parametrize("condition", ["grpo", "gigpo_dynamic", "unknown"])
def test_out_of_scope_conditions_are_refused(env, condition):
    r = subprocess.run(["bash", "scripts/run_condition.sh", condition], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, **env})
    assert r.returncode != 0 and "Unknown condition" in r.stderr


def test_nonzero_seed_is_refused(env):
    r = subprocess.run(["bash", "scripts/run_condition.sh", "gigpo"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, **env, "SEED": "1"})
    assert r.returncode == 2 and "Only SEED=0" in r.stderr


def test_submit_chain_rejects_out_of_scope_before_sbatch(tmp_path):
    for condition, seed in [("grpo", "0"), ("gigpo_dynamic", "0"), ("gigpo", "1")]:
        r = subprocess.run(["bash", "cluster/omni/submit_chain.sh", condition, "1"], cwd=ROOT,
                           capture_output=True, text=True, env={**os.environ, "SEED": seed})
        assert r.returncode == 2, (condition, seed, r.stdout, r.stderr)
        assert "Only" in r.stderr
