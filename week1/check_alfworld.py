"""Reset one ALFWorld text game and take one admissible action."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "upstream-verl-agent"
sys.path.insert(0, str(REPO))

from agent_system.environments.env_package.alfworld.envs import build_alfworld_envs  # noqa: E402


def main() -> None:
    config = REPO / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    env = build_alfworld_envs(str(config), seed=0, env_num=1, group_n=1, is_train=True)
    try:
        observations, _, infos = env.reset()
        commands = infos[0]["admissible_commands"]
        if not observations or not commands:
            raise RuntimeError("ALFWorld reset returned no observation or admissible action")
        action = commands[0]
        next_observations, _, rewards, dones, _ = env.step([action])
        if not next_observations:
            raise RuntimeError("ALFWorld step returned no observation")
        print("ALFWorld reset and step: PASS")
        print(f"Initial observation: {observations[0][:160]!r}")
        print(f"Action: {action!r}")
        print(f"Next observation: {next_observations[0][:160]!r}")
        print(f"Reward: {rewards[0]}, done: {dones[0]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
