from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

import openarm_env
from openarm_env.common.joint_controller import OpenArmJointPositionController
from openarm_env.pick_place.envs import OpenArmBimanualPickPlaceEnv
from openarm_env.pick_place.policies import OpenArmEpisodeReplayJointTargetPolicy


def resolve_episode_path(path: Path) -> Path:
    if path.is_file():
        return path

    episode_files = sorted(path.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found under {path}")
    return episode_files[0]


def main() -> None:
    episode_path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/act_pick_place/episode_0000.npz")
    env_id = sys.argv[2] if len(sys.argv) > 2 else openarm_env.PICK_PLACE_ENV_ID
    render_mode_arg = sys.argv[3] if len(sys.argv) > 3 else "human"
    initial_robot_pose = sys.argv[4] if len(sys.argv) > 4 else "basic"
    render_mode = None if render_mode_arg.lower() == "none" else render_mode_arg

    episode_path = resolve_episode_path(episode_path_arg)
    with np.load(episode_path, allow_pickle=False) as episode_data:
        saved_seed = int(episode_data["seed"][0])
        saved_steps = int(episode_data["episode_length"][0])
        saved_success = int(episode_data["episode_success"][0])

    env_kwargs = {"render_mode": render_mode}
    if env_id == openarm_env.PICK_PLACE_ENV_ID:
        env_kwargs["sticky_grasp"] = True
        env_kwargs["initial_robot_pose"] = initial_robot_pose

    env = gym.make(env_id, **env_kwargs)
    unwrapped = env.unwrapped
    if not isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
        raise TypeError(
            f"Replay runner currently supports the bimanual pick-place env only, got {type(unwrapped)!r}"
        )

    policy = OpenArmEpisodeReplayJointTargetPolicy(episode_path)
    low_level_controller = OpenArmJointPositionController()

    print(f"episode file: {episode_path}")
    print(f"env id: {env_id}")
    print(f"render mode: {render_mode_arg}")
    print(f"initial robot pose: {initial_robot_pose}")
    print(f"saved seed: {saved_seed}")
    print(f"saved steps: {saved_steps}")
    print(f"saved success: {saved_success}")

    observation, info = env.reset(seed=saved_seed)
    print(f"observation shape: {observation.shape}")
    print(f"initial distance: {info['distance_to_goal']:.4f}")
    if "cube_height" in info:
        print(f"initial cube height: {info['cube_height']:.4f}")

    max_steps = min(saved_steps, env.spec.max_episode_steps if env.spec is not None else saved_steps)
    try:
        for step in range(max_steps):
            joint_targets = policy.act(env)
            action = low_level_controller.act(env, joint_targets)
            observation, reward, terminated, truncated, info = env.step(action)

            if step % 10 == 0 or info["is_success"]:
                extra = ""
                if "cube_height" in info:
                    extra = (
                        f" cube_z={info['cube_height']:.4f}"
                        f" cube_speed={info['cube_speed']:.4f}"
                        f" sticky={info.get('sticky_attached', 0.0):.0f}:{info.get('sticky_side', '')}"
                    )
                print(
                    f"step={step:03d} reward={reward:.4f} "
                    f"distance={info['distance_to_goal']:.4f} success={info['is_success']:.0f}"
                    f"{extra}"
                )

            if terminated or truncated:
                print(
                    f"final_step={step:03d} final_distance={info['distance_to_goal']:.4f} "
                    f"success={info['is_success']:.0f}"
                )
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
