from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "inspect_act_dataset.py requires matplotlib. Install it in your environment first."
    ) from exc


CAMERA_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
JOINT_INDICES = np.arange(1, 19)
JOINT_TICK_LABELS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8\nleft_finger1",
    "9\nleft_finger2",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17\nright_finger1",
    "18\nright_finger2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one saved ACT episode with the three camera feeds plus "
            "robot state and action traces."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an episode .npz file or a dataset directory containing episode_XXXX.npz files.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to open when `path` is a directory.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Initial frame index.",
    )
    parser.add_argument(
        "--camera",
        choices=("all", "head", "wrist"),
        default="all",
        help="Display all cameras, only the head camera, or only the wrist cameras.",
    )
    return parser.parse_args()


def resolve_episode_path(path: Path, episode_index: int) -> Path:
    if path.is_file():
        return path

    episode_files = sorted(path.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found under {path}")
    if episode_index < 0 or episode_index >= len(episode_files):
        raise IndexError(
            f"episode-index {episode_index} is out of range for {len(episode_files)} episode files"
        )
    return episode_files[episode_index]


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def summarize_episode(path: Path, episode: dict[str, np.ndarray]) -> None:
    steps = int(episode["episode_length"][0])
    success = int(episode["episode_success"][0])
    seed = int(episode["seed"][0])
    print(f"episode file: {path}")
    print(f"steps: {steps}")
    print(f"seed: {seed}")
    print(f"success: {success}")
    print(f"initial distance: {float(episode['initial_distance_to_goal'][0]):.4f}")
    print(f"final distance: {float(episode['final_distance_to_goal'][0]):.4f}")
    print("keys:")
    for key in sorted(episode):
        print(f"  {key}: shape={episode[key].shape} dtype={episode[key].dtype}")


def setup_axes(camera_mode: str):
    if camera_mode == "head":
        figure = plt.figure(figsize=(14, 8))
        grid = figure.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.3, wspace=0.25)
        image_axes = [figure.add_subplot(grid[0, 0:3])]
    elif camera_mode == "wrist":
        figure = plt.figure(figsize=(14, 8))
        grid = figure.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.3, wspace=0.25)
        image_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    else:
        figure = plt.figure(figsize=(16, 9))
        grid = figure.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.3, wspace=0.25)
        image_axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[0, 2]),
        ]

    qpos_ax = figure.add_subplot(grid[1, 0])
    qvel_ax = figure.add_subplot(grid[1, 1])
    action_ax = figure.add_subplot(grid[1, 2])
    return figure, image_axes, qpos_ax, qvel_ax, action_ax


def camera_layout(camera_mode: str) -> list[tuple[str, str]]:
    if camera_mode == "head":
        return [("head_rgb", "head_camera")]
    if camera_mode == "wrist":
        return [
            ("left_wrist_rgb", "left_wrist_camera"),
            ("right_wrist_rgb", "right_wrist_camera"),
        ]
    return [
        ("head_rgb", "head_camera"),
        ("left_wrist_rgb", "left_wrist_camera"),
        ("right_wrist_rgb", "right_wrist_camera"),
    ]


def configure_bar_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("joint index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(JOINT_INDICES)
    ax.set_xticklabels(JOINT_TICK_LABELS, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def set_axis_limits(ax, values: np.ndarray, pad_ratio: float = 0.1) -> None:
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        low -= 1.0
        high += 1.0
    pad = (high - low) * pad_ratio
    ax.set_ylim(low - pad, high + pad)


def main() -> None:
    args = parse_args()
    episode_path = resolve_episode_path(args.path, args.episode_index)
    episode = load_episode(episode_path)
    summarize_episode(episode_path, episode)

    num_steps = int(episode["head_rgb"].shape[0])
    if num_steps == 0:
        raise ValueError("Episode contains zero timesteps.")

    initial_frame = int(np.clip(args.frame, 0, num_steps - 1))
    figure, image_axes, qpos_ax, qvel_ax, action_ax = setup_axes(args.camera)
    layout = camera_layout(args.camera)

    image_artists = []
    for ax, (camera_key, title) in zip(image_axes, layout):
        artist = ax.imshow(episode[camera_key][initial_frame])
        ax.set_title(title)
        ax.axis("off")
        image_artists.append((artist, camera_key))

    qpos_bars = qpos_ax.bar(JOINT_INDICES, episode["robot_qpos"][initial_frame], width=0.8)
    configure_bar_axis(qpos_ax, "robot_qpos", "position")

    qvel_bars = qvel_ax.bar(JOINT_INDICES, episode["robot_qvel"][initial_frame], width=0.8)
    configure_bar_axis(qvel_ax, "robot_qvel", "velocity")

    action_bars = action_ax.bar(JOINT_INDICES, episode["action"][initial_frame], width=0.8)
    configure_bar_axis(action_ax, "action", "command")

    set_axis_limits(qpos_ax, episode["robot_qpos"])
    set_axis_limits(qvel_ax, episode["robot_qvel"])
    set_axis_limits(action_ax, episode["action"])

    slider_ax = figure.add_axes([0.18, 0.03, 0.64, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="frame",
        valmin=0,
        valmax=num_steps - 1,
        valinit=initial_frame,
        valstep=1,
    )

    def sticky_side_label(value: int) -> str:
        if value == 1:
            return "left"
        if value == 2:
            return "right"
        return "-"

    def update(frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        for artist, camera_key in image_artists:
            artist.set_data(episode[camera_key][frame_idx])

        for bar, value in zip(qpos_bars, episode["robot_qpos"][frame_idx]):
            bar.set_height(float(value))
        for bar, value in zip(qvel_bars, episode["robot_qvel"][frame_idx]):
            bar.set_height(float(value))
        for bar, value in zip(action_bars, episode["action"][frame_idx]):
            bar.set_height(float(value))

        reward = float(episode["reward"][frame_idx, 0])
        success = int(episode["is_success"][frame_idx, 0])
        sticky_attached = int(episode["sticky_attached"][frame_idx, 0])
        sticky_side = sticky_side_label(int(episode["sticky_side"][frame_idx, 0]))
        cube_pos = episode["cube_pos"][frame_idx]
        goal = episode["goal"][frame_idx]
        figure.suptitle(
            "frame={frame:04d}/{last:04d}  "
            "reward={reward:.3f}  success={success}  "
            "sticky={sticky_attached}:{sticky_side}  "
            "cube=({cx:.3f}, {cy:.3f}, {cz:.3f})  "
            "goal=({gx:.3f}, {gy:.3f}, {gz:.3f})".format(
                frame=frame_idx,
                last=num_steps - 1,
                reward=reward,
                success=success,
                sticky_attached=sticky_attached,
                sticky_side=sticky_side,
                cx=float(cube_pos[0]),
                cy=float(cube_pos[1]),
                cz=float(cube_pos[2]),
                gx=float(goal[0]),
                gy=float(goal[1]),
                gz=float(goal[2]),
            )
        )
        figure.canvas.draw_idle()

    def on_slider_change(value: float) -> None:
        update(int(value))

    def on_key(event) -> None:
        current = int(slider.val)
        if event.key == "right":
            slider.set_val(min(current + 1, num_steps - 1))
        elif event.key == "left":
            slider.set_val(max(current - 1, 0))
        elif event.key == "up":
            slider.set_val(min(current + 10, num_steps - 1))
        elif event.key == "down":
            slider.set_val(max(current - 10, 0))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(num_steps - 1)

    slider.on_changed(on_slider_change)
    figure.canvas.mpl_connect("key_press_event", on_key)
    update(initial_frame)
    plt.show()


if __name__ == "__main__":
    main()
