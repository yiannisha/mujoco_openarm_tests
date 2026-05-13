from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

import openarm_env
from openarm_env.common.joint_controller import (
    JointPositionTargets,
    OpenArmJointPositionController,
)
from openarm_env.pick_place.envs import OpenArmBimanualPickPlaceEnv
from openarm_env.pick_place.policies import (
    OpenArmBimanualPickPlaceGraspIKJointTargetPolicy,
    OpenArmBimanualPickPlaceIKJointTargetPolicy,
)
from openarm_env.reach.envs import OpenArmBimanualReachEnv
from openarm_env.reach.policies import OpenArmIKJointTargetPolicy


CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ACT-style in-sim rollouts with RGB from the head/wrist cameras, "
            "robot proprioception, and actions."
        )
    )
    parser.add_argument(
        "--env-id",
        default=openarm_env.PICK_PLACE_ENV_ID,
        help="Bimanual env to roll out. Defaults to the pick-place env.",
    )
    parser.add_argument(
        "--policy",
        default="auto",
        choices=("auto", "push", "grasp"),
        help="High-level policy to use for the pick-place env.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets") / "act_openarm",
    )
    parser.add_argument(
        "--sticky-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the sticky-grasp cheat when using the pick-place env.",
    )
    parser.add_argument(
        "--initial-robot-pose",
        choices=("basic", "table_ready"),
        default="basic",
        help="Initial robot reset pose for the pick-place env.",
    )
    return parser.parse_args()


def choose_policy(env, policy_name: str):
    unwrapped = env.unwrapped
    if isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
        if policy_name == "grasp":
            return OpenArmBimanualPickPlaceGraspIKJointTargetPolicy()
        if policy_name == "push":
            return OpenArmBimanualPickPlaceIKJointTargetPolicy()
        return OpenArmBimanualPickPlaceGraspIKJointTargetPolicy()

    if isinstance(unwrapped, OpenArmBimanualReachEnv):
        return OpenArmIKJointTargetPolicy()

    raise TypeError(
        f"Dataset collection is currently implemented only for bimanual envs, got {type(unwrapped)!r}"
    )


def create_renderers(env, width: int, height: int) -> dict[str, mujoco.Renderer]:
    try:
        return {
            camera_name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            for camera_name in CAMERA_NAMES
        }
    except ValueError as exc:
        message = str(exc)
        if "framebuffer width" in message or "framebuffer height" in message:
            raise RuntimeError(
                "Requested camera resolution exceeds the MuJoCo offscreen framebuffer. "
                "Increase `<visual><global offwidth=... offheight=.../></visual>` in the scene XML, "
                f"or reduce `--width/--height`. Original error: {message.strip()}"
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            "Failed to create MuJoCo offscreen renderers. "
            "Run this from a local session with working OpenGL / MuJoCo rendering."
        ) from exc


def render_frames(
    renderers: dict[str, mujoco.Renderer],
    data: mujoco.MjData,
) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name, renderer in renderers.items():
        renderer.update_scene(data, camera=camera_name)
        frames[camera_name] = renderer.render().copy()
    return frames


def robot_qpos_qvel(env) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "_right_finger_qpos_slice"):
        raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")
    robot_qpos_stop = unwrapped._right_finger_qpos_slice.stop
    robot_qvel_stop = unwrapped._right_finger_qpos_slice.stop
    return (
        unwrapped.data.qpos[:robot_qpos_stop].copy(),
        unwrapped.data.qvel[:robot_qvel_stop].copy(),
    )


def targets_to_arrays(targets: JointPositionTargets) -> dict[str, np.ndarray]:
    def maybe_array(value: np.ndarray | None, size: int) -> np.ndarray:
        if value is None:
            return np.full(size, np.nan, dtype=np.float32)
        return np.asarray(value, dtype=np.float32).copy()

    def maybe_scalar(value: float | None) -> np.float32:
        if value is None:
            return np.float32(np.nan)
        return np.float32(value)

    return {
        "single_arm_qpos": maybe_array(targets.single_arm_qpos, 7),
        "left_arm_qpos": maybe_array(targets.left_arm_qpos, 7),
        "right_arm_qpos": maybe_array(targets.right_arm_qpos, 7),
        "single_arm_finger_torque": np.array([maybe_scalar(targets.single_arm_finger_torque)]),
        "left_finger_target": np.array([maybe_scalar(targets.left_finger_target)]),
        "right_finger_target": np.array([maybe_scalar(targets.right_finger_target)]),
    }


def collect_episode(
    env,
    renderers: dict[str, mujoco.Renderer],
    policy,
    controller: OpenArmJointPositionController,
    seed: int,
) -> dict[str, np.ndarray]:
    observation, info = env.reset(seed=seed)
    initial_observation = observation.copy()
    max_steps = env.spec.max_episode_steps if env.spec is not None else 250
    unwrapped = env.unwrapped

    head_rgb: list[np.ndarray] = []
    left_wrist_rgb: list[np.ndarray] = []
    right_wrist_rgb: list[np.ndarray] = []
    robot_qpos_seq: list[np.ndarray] = []
    robot_qvel_seq: list[np.ndarray] = []
    goal_seq: list[np.ndarray] = []
    cube_pos_seq: list[np.ndarray] = []
    cube_quat_seq: list[np.ndarray] = []
    cube_linear_velocity_seq: list[np.ndarray] = []
    cube_angular_velocity_seq: list[np.ndarray] = []
    desired_left_arm_qpos_seq: list[np.ndarray] = []
    desired_right_arm_qpos_seq: list[np.ndarray] = []
    desired_left_finger_seq: list[np.ndarray] = []
    desired_right_finger_seq: list[np.ndarray] = []
    action_seq: list[np.ndarray] = []
    reward_seq: list[np.ndarray] = []
    success_seq: list[np.ndarray] = []
    terminated_seq: list[np.ndarray] = []
    truncated_seq: list[np.ndarray] = []
    sticky_attached_seq: list[np.ndarray] = []
    sticky_side_seq: list[np.ndarray] = []
    sim_time_seq: list[np.ndarray] = []

    initial_goal = getattr(unwrapped, "_goal", np.zeros(3, dtype=np.float64)).copy()
    initial_info = dict(info)

    for _step in range(max_steps):
        frames = render_frames(renderers, unwrapped.data)
        robot_qpos, robot_qvel = robot_qpos_qvel(unwrapped)
        cube_pos, cube_quat, cube_linear_velocity, cube_angular_velocity = unwrapped._get_cube_state()
        joint_targets = policy.act(env)
        target_arrays = targets_to_arrays(joint_targets)
        action = controller.act(env, joint_targets)

        head_rgb.append(frames["head_camera"])
        left_wrist_rgb.append(frames["left_wrist_camera"])
        right_wrist_rgb.append(frames["right_wrist_camera"])
        robot_qpos_seq.append(robot_qpos.astype(np.float32))
        robot_qvel_seq.append(robot_qvel.astype(np.float32))
        goal_seq.append(unwrapped._goal.copy().astype(np.float32))
        cube_pos_seq.append(cube_pos.astype(np.float32))
        cube_quat_seq.append(cube_quat.astype(np.float32))
        cube_linear_velocity_seq.append(cube_linear_velocity.astype(np.float32))
        cube_angular_velocity_seq.append(cube_angular_velocity.astype(np.float32))
        desired_left_arm_qpos_seq.append(target_arrays["left_arm_qpos"])
        desired_right_arm_qpos_seq.append(target_arrays["right_arm_qpos"])
        desired_left_finger_seq.append(target_arrays["left_finger_target"])
        desired_right_finger_seq.append(target_arrays["right_finger_target"])
        action_seq.append(action.astype(np.float32))
        sim_time_seq.append(np.array([unwrapped.data.time], dtype=np.float32))

        observation, reward, terminated, truncated, info = env.step(action)

        reward_seq.append(np.array([reward], dtype=np.float32))
        success_seq.append(np.array([info["is_success"]], dtype=np.float32))
        terminated_seq.append(np.array([float(terminated)], dtype=np.float32))
        truncated_seq.append(np.array([float(truncated)], dtype=np.float32))
        sticky_attached_seq.append(
            np.array([float(info.get("sticky_attached", 0.0))], dtype=np.float32)
        )
        sticky_side_value = 1 if info.get("sticky_side", "") == "left" else 2 if info.get("sticky_side", "") == "right" else 0
        sticky_side_seq.append(np.array([sticky_side_value], dtype=np.int8))

        if terminated or truncated or info["is_success"]:
            break

    final_robot_qpos, final_robot_qvel = robot_qpos_qvel(unwrapped)
    final_cube_pos, final_cube_quat, final_cube_linear_velocity, final_cube_angular_velocity = (
        unwrapped._get_cube_state()
    )

    return {
        "head_rgb": np.stack(head_rgb, axis=0),
        "left_wrist_rgb": np.stack(left_wrist_rgb, axis=0),
        "right_wrist_rgb": np.stack(right_wrist_rgb, axis=0),
        "robot_qpos": np.stack(robot_qpos_seq, axis=0),
        "robot_qvel": np.stack(robot_qvel_seq, axis=0),
        "goal": np.stack(goal_seq, axis=0),
        "cube_pos": np.stack(cube_pos_seq, axis=0),
        "cube_quat": np.stack(cube_quat_seq, axis=0),
        "cube_linear_velocity": np.stack(cube_linear_velocity_seq, axis=0),
        "cube_angular_velocity": np.stack(cube_angular_velocity_seq, axis=0),
        "desired_left_arm_qpos": np.stack(desired_left_arm_qpos_seq, axis=0),
        "desired_right_arm_qpos": np.stack(desired_right_arm_qpos_seq, axis=0),
        "desired_left_finger_target": np.stack(desired_left_finger_seq, axis=0),
        "desired_right_finger_target": np.stack(desired_right_finger_seq, axis=0),
        "action": np.stack(action_seq, axis=0),
        "reward": np.stack(reward_seq, axis=0),
        "is_success": np.stack(success_seq, axis=0),
        "terminated": np.stack(terminated_seq, axis=0),
        "truncated": np.stack(truncated_seq, axis=0),
        "sticky_attached": np.stack(sticky_attached_seq, axis=0),
        "sticky_side": np.stack(sticky_side_seq, axis=0),
        "sim_time": np.stack(sim_time_seq, axis=0),
        "initial_observation": initial_observation.astype(np.float32),
        "initial_goal": initial_goal.astype(np.float32),
        "final_robot_qpos": final_robot_qpos.astype(np.float32),
        "final_robot_qvel": final_robot_qvel.astype(np.float32),
        "final_cube_pos": final_cube_pos.astype(np.float32),
        "final_cube_quat": final_cube_quat.astype(np.float32),
        "final_cube_linear_velocity": final_cube_linear_velocity.astype(np.float32),
        "final_cube_angular_velocity": final_cube_angular_velocity.astype(np.float32),
        "episode_length": np.array([len(action_seq)], dtype=np.int32),
        "episode_success": np.array([float(info["is_success"])], dtype=np.float32),
        "seed": np.array([seed], dtype=np.int32),
        "initial_distance_to_goal": np.array([initial_info["distance_to_goal"]], dtype=np.float32),
        "final_distance_to_goal": np.array([info["distance_to_goal"]], dtype=np.float32),
    }


def json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: json_safe_value(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: json_safe_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    env,
    policy,
    controller: OpenArmJointPositionController,
) -> None:
    manifest = {
        "env_id": args.env_id,
        "policy_name": policy.__class__.__name__,
        "controller_name": controller.__class__.__name__,
        "episodes": args.episodes,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "cameras": list(CAMERA_NAMES),
        "sticky_grasp": args.sticky_grasp,
        "initial_robot_pose": args.initial_robot_pose,
        "max_episode_steps": env.spec.max_episode_steps if env.spec is not None else None,
        "policy_config": json_safe_value(policy),
        "controller_config": json_safe_value(controller),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs: dict[str, Any] = {
        "render_mode": None,
        "width": args.width,
        "height": args.height,
    }
    if args.env_id == openarm_env.PICK_PLACE_ENV_ID:
        env_kwargs["sticky_grasp"] = args.sticky_grasp
        env_kwargs["initial_robot_pose"] = args.initial_robot_pose

    env = gym.make(args.env_id, **env_kwargs)
    policy = choose_policy(env, args.policy)
    controller = OpenArmJointPositionController()
    renderers = create_renderers(env, args.width, args.height)
    write_manifest(args.output_dir, args, env, policy, controller)

    print(f"env id: {args.env_id}")
    print(f"episodes: {args.episodes}")
    print(f"policy: {policy.__class__.__name__}")
    print(f"output_dir: {args.output_dir.resolve()}")

    try:
        for episode_idx in range(args.episodes):
            seed = args.seed + episode_idx
            episode_data = collect_episode(
                env=env,
                renderers=renderers,
                policy=policy,
                controller=controller,
                seed=seed,
            )
            episode_path = args.output_dir / f"episode_{episode_idx:04d}.npz"
            np.savez_compressed(episode_path, **episode_data)
            print(
                f"episode={episode_idx:04d} "
                f"steps={int(episode_data['episode_length'][0])} "
                f"success={int(episode_data['episode_success'][0])} "
                f"final_distance={float(episode_data['final_distance_to_goal'][0]):.4f} "
                f"saved={episode_path.name}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
