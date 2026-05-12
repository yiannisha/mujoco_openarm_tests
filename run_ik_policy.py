from __future__ import annotations

import sys

import gymnasium as gym

import openarm_env
from openarm_env.common.joint_controller import OpenArmJointPositionController
from openarm_env.pick_place.envs import OpenArmBimanualPickPlaceEnv
from openarm_env.pick_place.policies import (
    OpenArmBimanualPickPlaceGraspIKJointTargetPolicy,
    OpenArmBimanualPickPlaceIKJointTargetPolicy,
)
from openarm_env.reach.policies import OpenArmIKJointTargetPolicy


def main() -> None:
    env_id = sys.argv[1] if len(sys.argv) > 1 else openarm_env.BIMANUAL_ENV_ID
    render_mode_arg = sys.argv[2] if len(sys.argv) > 2 else "human"
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    policy_name = sys.argv[4].lower() if len(sys.argv) > 4 else "auto"
    render_mode = None if render_mode_arg.lower() == "none" else render_mode_arg

    env = gym.make(env_id, render_mode=render_mode)
    unwrapped = env.unwrapped
    if isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
        if policy_name == "grasp":
            policy = OpenArmBimanualPickPlaceGraspIKJointTargetPolicy()
        else:
            policy = OpenArmBimanualPickPlaceIKJointTargetPolicy()
    else:
        policy = OpenArmIKJointTargetPolicy()
    low_level_controller = OpenArmJointPositionController()

    print(f"env id: {env_id}")
    print(f"episodes: {num_episodes}")
    print(f"policy: {policy.__class__.__name__}")
    max_steps = env.spec.max_episode_steps if env.spec is not None else 250

    for episode in range(num_episodes):
        observation, info = env.reset(seed=69 + episode)
        print(f"episode={episode:02d} observation shape={observation.shape}")
        print(f"episode={episode:02d} initial distance={info['distance_to_goal']:.4f}")
        if "cube_height" in info:
            print(f"episode={episode:02d} initial cube height={info['cube_height']:.4f}")

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
                    )
                print(
                    f"episode={episode:02d} step={step:03d} reward={reward:.4f} "
                    f"distance={info['distance_to_goal']:.4f} success={info['is_success']:.0f}"
                    f"{extra}"
                )

            if terminated or truncated or info["is_success"]:
                print(
                    f"episode={episode:02d} final_step={step:03d} "
                    f"final_distance={info['distance_to_goal']:.4f} "
                    f"success={info['is_success']:.0f}"
                )
                break

    env.close()


if __name__ == "__main__":
    main()
