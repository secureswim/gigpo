"""Check shared settings in the pinned GiGPO and GRPO ALFWorld scripts."""

from pathlib import Path
import re
import sys
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "upstream-verl-agent"
EXPECTED_SHA = "f974dc5977908d6007ad178c9f3c48f0c7b39331"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assignments(script: str) -> dict[str, str]:
    values = {}
    for line in script.splitlines():
        line = line.strip().rstrip("\\").strip()
        match = re.match(r"([A-Za-z_][\w.]*)=(.+)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def main() -> int:
    sha = (REPO / "UPSTREAM_COMMIT").read_text(encoding="utf-8-sig").strip()
    if sha != EXPECTED_SHA:
        print(f"Wrong upstream commit: {sha}; expected {EXPECTED_SHA}", file=sys.stderr)
        return 1

    for line in (ROOT / "UPSTREAM_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected_digest, relative_path = line.split("  ", 1)
        actual_digest = sha256((REPO / relative_path).read_bytes()).hexdigest()
        # Original manifest was generated with CRLF; Git may check out LF.
        canonical_crlf = (REPO / relative_path).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        crlf_digest = sha256(canonical_crlf).hexdigest()
        if expected_digest not in (actual_digest, crlf_digest):
            print(f"Upstream snapshot differs: {relative_path}", file=sys.stderr)
            return 1

    gigpo = assignments(source(REPO / "examples/gigpo_trainer/run_alfworld.sh"))
    grpo = assignments(source(REPO / "examples/grpo_trainer/run_alfworld.sh"))
    shared = [
        "train_data_size", "val_data_size", "group_size",
        "data.train_batch_size", "data.val_batch_size",
        "actor_rollout_ref.model.path", "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.invalid_action_penalty_coef",
        "actor_rollout_ref.rollout.val_kwargs.temperature",
        "actor_rollout_ref.rollout.val_kwargs.do_sample",
        "env.env_name", "env.seed", "env.max_steps", "env.rollout.n",
        "trainer.n_gpus_per_node", "trainer.test_freq", "trainer.total_epochs",
    ]
    differences = [(key, gigpo.get(key), grpo.get(key)) for key in shared
                   if gigpo.get(key) is None or gigpo.get(key) != grpo.get(key)]
    if differences:
        for key, a, b in differences:
            print(f"MISMATCH {key}: GiGPO={a!r}, GRPO={b!r}", file=sys.stderr)
        return 1

    if gigpo.get("algorithm.adv_estimator") != "gigpo" or grpo.get("algorithm.adv_estimator") != "grpo":
        print("Algorithm selectors are unexpected", file=sys.stderr)
        return 1
    print(f"Pinned upstream commit: {sha}")
    print(f"Shared ALFWorld settings match ({len(shared)} checked).")
    print("Method-specific settings: GiGPO adds gamma=0.95, step weight=1.0, mode=mean_std_norm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
