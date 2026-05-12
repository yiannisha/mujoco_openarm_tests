from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image

import openarm_env
from openarm_env.ik_policy import OpenArmIKJointTargetPolicy
from openarm_env.joint_controller import OpenArmJointPositionController

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the OpenArm wrist camera feeds after editing the XML camera poses."
    )
    parser.add_argument("--env-id", default=openarm_env.BIMANUAL_ENV_ID)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--pause", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camera_previews"),
        help="Used only when matplotlib is unavailable.",
    )
    return parser.parse_args()


def combine_frames(left_frame: np.ndarray, right_frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.concatenate((left_frame, right_frame), axis=1))


def render_camera_pair(
    renderers: dict[str, mujoco.Renderer],
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    renderers["left_wrist_camera"].update_scene(data, camera="left_wrist_camera")
    renderers["right_wrist_camera"].update_scene(data, camera="right_wrist_camera")
    left_frame = renderers["left_wrist_camera"].render()
    right_frame = renderers["right_wrist_camera"].render()
    return left_frame, right_frame


def main() -> None:
    args = parse_args()

    env = gym.make(args.env_id, render_mode=None)
    policy = OpenArmIKJointTargetPolicy()
    controller = OpenArmJointPositionController()
    try:
        renderers = {
            "left_wrist_camera": mujoco.Renderer(
                env.unwrapped.model, height=args.height, width=args.width
            ),
            "right_wrist_camera": mujoco.Renderer(
                env.unwrapped.model, height=args.height, width=args.width
            ),
        }
    except Exception as exc:
        env.close()
        raise RuntimeError(
            "Failed to create the MuJoCo offscreen renderer. "
            "Run this script from a local desktop session with OpenGL support."
        ) from exc

    if plt is not None:
        plt.ion()
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        left_artist = axes[0].imshow([[0]], animated=False)
        right_artist = axes[1].imshow([[0]], animated=False)
        axes[0].axis("off")
        axes[1].axis("off")
        axes[0].set_title("left_wrist_camera")
        axes[1].set_title("right_wrist_camera")
    else:
        figure = None
        axes = None
        left_artist = None
        right_artist = None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"matplotlib not available; saving previews to {args.output_dir.resolve()}"
        )

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(
                f"episode={episode:02d} observation_shape={observation.shape} "
                f"initial_distance={info['distance_to_goal']:.4f}"
            )

            saved_frames: list[Image.Image] = []
            final_combined_frame: Image.Image | None = None

            for step in range(args.steps):
                joint_targets = policy.act(env)
                action = controller.act(env, joint_targets)
                observation, reward, terminated, truncated, info = env.step(action)

                left_frame, right_frame = render_camera_pair(renderers, env.unwrapped.data)
                final_combined_frame = combine_frames(left_frame, right_frame)

                if plt is not None:
                    left_artist.set_data(left_frame)
                    right_artist.set_data(right_frame)
                    axes[0].set_title(
                        f"left_wrist_camera\nstep={step:03d} distance={info['distance_to_goal']:.4f}"
                    )
                    axes[1].set_title(
                        f"right_wrist_camera\nstep={step:03d} distance={info['distance_to_goal']:.4f}"
                    )
                    figure.canvas.draw_idle()
                    plt.pause(args.pause)
                else:
                    saved_frames.append(final_combined_frame.copy())

                if step % 10 == 0 or info["is_success"]:
                    print(
                        f"episode={episode:02d} step={step:03d} reward={reward:.4f} "
                        f"distance={info['distance_to_goal']:.4f} success={info['is_success']:.0f}"
                    )

                if terminated or truncated or info["is_success"]:
                    print(
                        f"episode={episode:02d} final_step={step:03d} "
                        f"final_distance={info['distance_to_goal']:.4f} "
                        f"success={info['is_success']:.0f}"
                    )
                    break

            if plt is None and final_combined_frame is not None and saved_frames:
                preview_png = args.output_dir / f"episode_{episode:02d}_wrist_preview.png"
                preview_gif = args.output_dir / f"episode_{episode:02d}_wrist_preview.gif"
                final_combined_frame.save(preview_png)
                saved_frames[0].save(
                    preview_gif,
                    save_all=True,
                    append_images=saved_frames[1:],
                    duration=60,
                    loop=0,
                )
                print(f"saved {preview_png}")
                print(f"saved {preview_gif}")

        if plt is not None:
            print("Close the matplotlib window or press Ctrl+C to exit.")
            plt.ioff()
            plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
