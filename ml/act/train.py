from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SummaryWriter = None

from ml.act.model import (
    OPENARM_ACTION_DIM,
    OPENARM_QPOS_DIM,
    OpenArmACTPolicy,
    JointPositionTargets,
    load_openarm_act_checkpoint,
    save_openarm_act_checkpoint,
)


DEFAULT_CAMERAS = ["head_camera", "left_wrist_camera", "right_wrist_camera"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def targets_to_openarm_action_vector(targets: JointPositionTargets) -> np.ndarray:
    if targets.left_arm_qpos is None or targets.right_arm_qpos is None:
        raise ValueError("Bimanual eval targets must include left and right arm qpos")
    left_finger = 0.02 if targets.left_finger_target is None else targets.left_finger_target
    right_finger = 0.02 if targets.right_finger_target is None else targets.right_finger_target
    return np.concatenate(
        (
            np.asarray(targets.left_arm_qpos, dtype=np.float32),
            np.array([left_finger], dtype=np.float32),
            np.asarray(targets.right_arm_qpos, dtype=np.float32),
            np.array([right_finger], dtype=np.float32),
        )
    )


def discover_successful_episodes(dataset_dir: Path) -> list[Path]:
    episode_paths = sorted(dataset_dir.glob("episode_*.npz"))
    successful_paths: list[Path] = []
    for path in episode_paths:
        with np.load(path) as data:
            if "episode_success" in data:
                success = bool(float(np.asarray(data["episode_success"]).reshape(-1)[0]) == 1.0)
            elif "is_success" in data:
                success = bool(np.asarray(data["is_success"]).reshape(-1).max() == 1.0)
            else:
                success = False
        if success:
            successful_paths.append(path)
    if not successful_paths:
        raise RuntimeError(f"No successful .npz episodes found in {dataset_dir}")
    return successful_paths


def target_chunk_from_episode(data: np.lib.npyio.NpzFile) -> np.ndarray:
    qpos = np.asarray(data["robot_qpos"], dtype=np.float32)
    left_arm = np.asarray(data["desired_left_arm_qpos"], dtype=np.float32)
    left_finger = np.asarray(data["desired_left_finger_target"], dtype=np.float32).reshape(-1, 1)
    right_arm = np.asarray(data["desired_right_arm_qpos"], dtype=np.float32)
    right_finger = np.asarray(data["desired_right_finger_target"], dtype=np.float32).reshape(-1, 1)

    fallback_left_finger = qpos[:, 7:9].mean(axis=1, keepdims=True)
    fallback_right_finger = qpos[:, 16:18].mean(axis=1, keepdims=True)
    left_arm = np.where(np.isfinite(left_arm), left_arm, qpos[:, 0:7])
    left_finger = np.where(np.isfinite(left_finger), left_finger, fallback_left_finger)
    right_arm = np.where(np.isfinite(right_arm), right_arm, qpos[:, 9:16])
    right_finger = np.where(np.isfinite(right_finger), right_finger, fallback_right_finger)

    return np.concatenate((left_arm, left_finger, right_arm, right_finger), axis=1).astype(np.float32)


def compute_norm_stats(paths: list[Path]) -> dict[str, np.ndarray]:
    qpos_parts = []
    action_parts = []
    for path in paths:
        with np.load(path) as data:
            qpos = np.asarray(data["robot_qpos"], dtype=np.float32)
            action = target_chunk_from_episode(data)
        qpos_parts.append(qpos)
        action_parts.append(action)

    qpos_all = np.concatenate(qpos_parts, axis=0)
    action_all = np.concatenate(action_parts, axis=0)
    qpos_std = np.clip(qpos_all.std(axis=0), 1e-2, np.inf)
    action_std = np.clip(action_all.std(axis=0), 1e-2, np.inf)
    return {
        "qpos_mean": qpos_all.mean(axis=0).astype(np.float32),
        "qpos_std": qpos_std.astype(np.float32),
        "action_mean": action_all.mean(axis=0).astype(np.float32),
        "action_std": action_std.astype(np.float32),
    }


class OpenArmACTEpisodeDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        chunk_size: int,
    ) -> None:
        self.paths = paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        with np.load(path) as data:
            qpos_seq = np.asarray(data["robot_qpos"], dtype=np.float32)
            action_seq = target_chunk_from_episode(data)
            episode_len = min(len(qpos_seq), len(action_seq))
            start_ts = int(torch.randint(episode_len, (1,)).item())

            images = []
            for camera_name in self.camera_names:
                key = f"{camera_name.replace('_camera', '')}_rgb"
                if key not in data:
                    key = f"{camera_name}_rgb"
                if key not in data:
                    raise KeyError(f"{path} does not contain RGB frames for {camera_name!r}")
                images.append(np.asarray(data[key][start_ts]))

            qpos = qpos_seq[start_ts]
            future = action_seq[start_ts : start_ts + self.chunk_size]

        action_len = len(future)
        padded_action = np.zeros((self.chunk_size, OPENARM_ACTION_DIM), dtype=np.float32)
        padded_action[:action_len] = future
        is_pad = np.ones((self.chunk_size,), dtype=bool)
        is_pad[:action_len] = False

        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        padded_action = (
            (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        ).astype(np.float32)
        padded_action[is_pad] = 0.0

        image = np.stack(images, axis=0)
        image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0
        qpos_tensor = torch.from_numpy(qpos.astype(np.float32))
        action_tensor = torch.from_numpy(padded_action)
        is_pad_tensor = torch.from_numpy(is_pad)
        return image_tensor, qpos_tensor, action_tensor, is_pad_tensor


def split_episodes(paths: list[Path], seed: int, val_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    if not train_paths:
        train_paths = val_paths
    return train_paths, val_paths


def mean_loss_dict(loss_dicts: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    return {
        key: float(torch.stack([loss_dict[key].detach().cpu() for loss_dict in loss_dicts]).mean())
        for key in loss_dicts[0]
    }


def log_scalar_dict(writer: Any, prefix: str, metrics: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, dict):
            log_scalar_dict(writer, f"{prefix}/{key}", value, step)
        elif isinstance(value, (float, int, np.floating, np.integer)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def forward_batch(
    policy: OpenArmACTPolicy,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image, qpos, action, is_pad = batch
    return policy(
        qpos.to(device, non_blocking=True),
        image.to(device, non_blocking=True),
        action.to(device, non_blocking=True),
        is_pad.to(device, non_blocking=True),
    )


def train_policy(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.tensorboard_dir or (args.ckpt_dir / "tensorboard")
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("TensorBoard logging disabled because torch.utils.tensorboard is unavailable.")

    successful_paths = discover_successful_episodes(args.dataset_dir)
    train_paths, val_paths = split_episodes(successful_paths, args.seed)
    norm_stats = compute_norm_stats(train_paths)

    train_dataset = OpenArmACTEpisodeDataset(train_paths, args.camera_names, norm_stats, args.chunk_size)
    val_dataset = OpenArmACTEpisodeDataset(val_paths, args.camera_names, norm_stats, args.chunk_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    policy_config = {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "backbone": args.backbone,
        "pretrained_backbone": args.pretrained_backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": args.camera_names,
        "qpos_dim": OPENARM_QPOS_DIM,
        "action_dim": OPENARM_ACTION_DIM,
    }
    policy = OpenArmACTPolicy(policy_config, norm_stats=norm_stats, device=device, temporal_agg=args.temporal_agg)
    optimizer = policy.configure_optimizer()

    metadata = {
        "dataset_dir": args.dataset_dir,
        "successful_episodes": len(successful_paths),
        "train_episodes": [path.name for path in train_paths],
        "val_episodes": [path.name for path in val_paths],
        "camera_names": args.camera_names,
        "qpos_dim": OPENARM_QPOS_DIM,
        "action_dim": OPENARM_ACTION_DIM,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "device": str(device),
    }
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2))
    (args.ckpt_dir / "norm_stats.json").write_text(json.dumps(json_safe(norm_stats), indent=2))
    writer.add_text("metadata/run", json.dumps(json_safe(metadata), indent=2), 0)
    writer.add_text("metadata/norm_stats", json.dumps(json_safe(norm_stats), indent=2), 0)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    try:
        for epoch in range(args.num_epochs):
            policy.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss_dict = forward_batch(policy, batch, device)
                loss_dict["loss"].backward()
                optimizer.step()
                train_losses.append(loss_dict)
            train_summary = mean_loss_dict(train_losses)

            policy.eval()
            val_losses = []
            with torch.inference_mode():
                for batch in val_loader:
                    val_losses.append(forward_batch(policy, batch, device))
            val_summary = mean_loss_dict(val_losses)

            epoch_metrics = {"epoch": epoch, "train": train_summary, "val": val_summary}
            history.append(epoch_metrics)
            (args.ckpt_dir / "metrics.json").write_text(json.dumps(json_safe(history), indent=2))
            log_scalar_dict(writer, "train", train_summary, epoch)
            log_scalar_dict(writer, "val", val_summary, epoch)
            if writer is not None:
                writer.add_scalar("epoch", epoch, epoch)

            val_loss = val_summary["loss"]
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_summary['loss']:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_l1={val_summary['l1']:.6f} "
                f"val_kl={val_summary['kl']:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(policy.state_dict())
                save_openarm_act_checkpoint(
                    args.ckpt_dir / "policy_best.ckpt",
                    policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_openarm_act_checkpoint(
                    args.ckpt_dir / f"policy_epoch_{epoch:04d}.ckpt",
                    policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            if args.eval_every > 0 and args.eval_rollouts > 0 and (epoch + 1) % args.eval_every == 0:
                eval_tag = f"epoch_{epoch:04d}_best_epoch_{best_epoch:04d}"
                eval_metrics = evaluate_checkpoint(
                    args, args.ckpt_dir / "policy_best.ckpt", eval_tag=eval_tag
                )
                epoch_metrics["eval"] = eval_metrics
                (args.ckpt_dir / "metrics.json").write_text(json.dumps(json_safe(history), indent=2))
                log_scalar_dict(writer, "eval", eval_metrics, epoch)
                policy.train()

        save_openarm_act_checkpoint(
            args.ckpt_dir / "policy_last.ckpt",
            policy,
            optimizer=optimizer,
            epoch=args.num_epochs - 1,
            metrics=history[-1] if history else {},
        )
        if best_state is not None:
            policy.load_state_dict(best_state)
            save_openarm_act_checkpoint(
                args.ckpt_dir / f"policy_epoch_{best_epoch:04d}_best.ckpt",
                policy,
                optimizer=optimizer,
                epoch=best_epoch,
                metrics={"best_val_loss": best_val_loss, "best_epoch": best_epoch},
            )

        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "metadata": metadata,
        }
        (args.ckpt_dir / "train_summary.json").write_text(json.dumps(json_safe(summary), indent=2))
        if writer is not None:
            writer.add_scalar("best_val_loss", best_val_loss, best_epoch if best_epoch >= 0 else 0)
        return args.ckpt_dir / "policy_best.ckpt", summary
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def create_renderers(env, camera_names: list[str], width: int, height: int) -> dict[str, Any]:
    import mujoco

    return {
        camera_name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
        for camera_name in camera_names
    }


def render_frames(renderers: dict[str, Any], env) -> dict[str, np.ndarray]:
    frames = {}
    for camera_name, renderer in renderers.items():
        renderer.update_scene(env.unwrapped.data, camera=camera_name)
        frames[camera_name] = renderer.render().copy()
    return frames


def eval_camera_names(policy_camera_names: list[str]) -> list[str]:
    names = list(dict.fromkeys([*DEFAULT_CAMERAS, *policy_camera_names]))
    return names


def robot_qpos_qvel_from_env(env) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "_right_finger_qpos_slice"):
        raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")
    robot_qpos_stop = unwrapped._right_finger_qpos_slice.stop
    robot_qvel_stop = unwrapped._right_finger_qpos_slice.stop
    return (
        unwrapped.data.qpos[:robot_qpos_stop].astype(np.float32).copy(),
        unwrapped.data.qvel[:robot_qvel_stop].astype(np.float32).copy(),
    )


def sticky_side_code(info: dict[str, Any]) -> int:
    sticky_side = info.get("sticky_side", "")
    if sticky_side == "left":
        return 1
    if sticky_side == "right":
        return 2
    return 0


def save_eval_episode(
    path: Path,
    rollout: dict[str, list[np.ndarray]],
    initial_observation: np.ndarray,
    initial_goal: np.ndarray,
    initial_info: dict[str, Any],
    final_info: dict[str, Any],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def stack(name: str) -> np.ndarray:
        values = rollout[name]
        if not values:
            raise ValueError(f"Cannot save empty rollout field {name!r}")
        return np.stack(values, axis=0)

    np.savez_compressed(
        path,
        head_rgb=stack("head_rgb"),
        left_wrist_rgb=stack("left_wrist_rgb"),
        right_wrist_rgb=stack("right_wrist_rgb"),
        robot_qpos=stack("robot_qpos"),
        robot_qvel=stack("robot_qvel"),
        goal=stack("goal"),
        cube_pos=stack("cube_pos"),
        cube_quat=stack("cube_quat"),
        cube_linear_velocity=stack("cube_linear_velocity"),
        cube_angular_velocity=stack("cube_angular_velocity"),
        desired_left_arm_qpos=stack("desired_left_arm_qpos"),
        desired_right_arm_qpos=stack("desired_right_arm_qpos"),
        desired_left_finger_target=stack("desired_left_finger_target"),
        desired_right_finger_target=stack("desired_right_finger_target"),
        predicted_joint_target=stack("predicted_joint_target"),
        action=stack("action"),
        reward=stack("reward"),
        is_success=stack("is_success"),
        terminated=stack("terminated"),
        truncated=stack("truncated"),
        sticky_attached=stack("sticky_attached"),
        sticky_side=stack("sticky_side"),
        sim_time=stack("sim_time"),
        initial_observation=np.asarray(initial_observation, dtype=np.float32),
        initial_goal=np.asarray(initial_goal, dtype=np.float32),
        final_robot_qpos=rollout["final_robot_qpos"][0],
        final_robot_qvel=rollout["final_robot_qvel"][0],
        final_cube_pos=rollout["final_cube_pos"][0],
        final_cube_quat=rollout["final_cube_quat"][0],
        final_cube_linear_velocity=rollout["final_cube_linear_velocity"][0],
        final_cube_angular_velocity=rollout["final_cube_angular_velocity"][0],
        episode_length=np.array([len(rollout["action"])], dtype=np.int32),
        episode_success=np.array([float(final_info.get("is_success", 0.0))], dtype=np.float32),
        seed=np.array([seed], dtype=np.int32),
        initial_distance_to_goal=np.array(
            [float(initial_info.get("distance_to_goal", np.nan))], dtype=np.float32
        ),
        final_distance_to_goal=np.array(
            [float(final_info.get("distance_to_goal", np.nan))], dtype=np.float32
        ),
    )


def evaluate_checkpoint(
    args: argparse.Namespace,
    ckpt_path: Path,
    eval_tag: str | None = None,
) -> dict[str, Any]:
    if args.eval_rollouts <= 0:
        return {"rollouts": 0, "success_rate": None, "mean_final_distance": None, "episodes": []}

    import gymnasium as gym

    import openarm_env
    from openarm_env.common.joint_controller import OpenArmJointPositionController

    device = resolve_device(args.device)
    policy, _checkpoint = load_openarm_act_checkpoint(
        ckpt_path,
        device=device,
        temporal_agg=args.temporal_agg,
    )
    controller = OpenArmJointPositionController()
    env = gym.make(
        openarm_env.PICK_PLACE_ENV_ID,
        render_mode=None,
        width=args.eval_width,
        height=args.eval_height,
        sticky_grasp=args.sticky_grasp,
    )
    camera_names = eval_camera_names(policy.camera_names)
    renderers = create_renderers(env, camera_names, args.eval_width, args.eval_height)
    max_steps = args.eval_max_steps or (env.spec.max_episode_steps if env.spec is not None else 450)
    tag = eval_tag or ckpt_path.stem
    rollout_dir = (args.eval_output_dir or (args.ckpt_dir / "eval_rollouts")) / tag

    episodes = []
    try:
        for rollout_idx in range(args.eval_rollouts):
            seed = args.seed + args.eval_seed_offset + rollout_idx
            initial_observation, info = env.reset(seed=seed)
            initial_info = dict(info)
            initial_goal = getattr(env.unwrapped, "_goal", np.zeros(3, dtype=np.float64)).copy()
            policy.reset()
            success = False
            final_distance = float(info.get("distance_to_goal", np.nan))
            steps = 0
            rollout: dict[str, list[np.ndarray]] = {
                "head_rgb": [],
                "left_wrist_rgb": [],
                "right_wrist_rgb": [],
                "robot_qpos": [],
                "robot_qvel": [],
                "goal": [],
                "cube_pos": [],
                "cube_quat": [],
                "cube_linear_velocity": [],
                "cube_angular_velocity": [],
                "desired_left_arm_qpos": [],
                "desired_right_arm_qpos": [],
                "desired_left_finger_target": [],
                "desired_right_finger_target": [],
                "predicted_joint_target": [],
                "action": [],
                "reward": [],
                "is_success": [],
                "terminated": [],
                "truncated": [],
                "sticky_attached": [],
                "sticky_side": [],
                "sim_time": [],
                "final_robot_qpos": [],
                "final_robot_qvel": [],
                "final_cube_pos": [],
                "final_cube_quat": [],
                "final_cube_linear_velocity": [],
                "final_cube_angular_velocity": [],
            }
            for step in range(max_steps):
                frames = render_frames(renderers, env)
                robot_qpos, robot_qvel = robot_qpos_qvel_from_env(env)
                cube_pos, cube_quat, cube_linear_velocity, cube_angular_velocity = (
                    env.unwrapped._get_cube_state()
                )
                targets = policy.act(env, frames)
                predicted_target = targets_to_openarm_action_vector(targets)
                action = controller.act(env, targets)
                if action.shape != env.action_space.shape:
                    raise RuntimeError(
                        f"Controller action shape {action.shape} does not match env action space {env.action_space.shape}"
                    )
                _observation, reward, terminated, truncated, info = env.step(action)
                steps = step + 1
                success = bool(info.get("is_success", 0.0))
                final_distance = float(info.get("distance_to_goal", np.nan))

                rollout["head_rgb"].append(frames["head_camera"])
                rollout["left_wrist_rgb"].append(frames["left_wrist_camera"])
                rollout["right_wrist_rgb"].append(frames["right_wrist_camera"])
                rollout["robot_qpos"].append(robot_qpos)
                rollout["robot_qvel"].append(robot_qvel)
                rollout["goal"].append(env.unwrapped._goal.astype(np.float32).copy())
                rollout["cube_pos"].append(cube_pos.astype(np.float32))
                rollout["cube_quat"].append(cube_quat.astype(np.float32))
                rollout["cube_linear_velocity"].append(cube_linear_velocity.astype(np.float32))
                rollout["cube_angular_velocity"].append(cube_angular_velocity.astype(np.float32))
                rollout["desired_left_arm_qpos"].append(predicted_target[0:7])
                rollout["desired_right_arm_qpos"].append(predicted_target[8:15])
                rollout["desired_left_finger_target"].append(predicted_target[7:8])
                rollout["desired_right_finger_target"].append(predicted_target[15:16])
                rollout["predicted_joint_target"].append(predicted_target)
                rollout["action"].append(action.astype(np.float32))
                rollout["reward"].append(np.array([reward], dtype=np.float32))
                rollout["is_success"].append(np.array([float(success)], dtype=np.float32))
                rollout["terminated"].append(np.array([float(terminated)], dtype=np.float32))
                rollout["truncated"].append(np.array([float(truncated)], dtype=np.float32))
                rollout["sticky_attached"].append(
                    np.array([float(info.get("sticky_attached", 0.0))], dtype=np.float32)
                )
                rollout["sticky_side"].append(np.array([sticky_side_code(info)], dtype=np.int8))
                rollout["sim_time"].append(np.array([env.unwrapped.data.time], dtype=np.float32))

                if terminated or truncated or success:
                    break

            final_robot_qpos, final_robot_qvel = robot_qpos_qvel_from_env(env)
            final_cube_pos, final_cube_quat, final_cube_linear_velocity, final_cube_angular_velocity = (
                env.unwrapped._get_cube_state()
            )
            rollout["final_robot_qpos"].append(final_robot_qpos)
            rollout["final_robot_qvel"].append(final_robot_qvel)
            rollout["final_cube_pos"].append(final_cube_pos.astype(np.float32))
            rollout["final_cube_quat"].append(final_cube_quat.astype(np.float32))
            rollout["final_cube_linear_velocity"].append(final_cube_linear_velocity.astype(np.float32))
            rollout["final_cube_angular_velocity"].append(final_cube_angular_velocity.astype(np.float32))

            episode_path = None
            if args.save_eval_rollouts:
                episode_path = rollout_dir / f"episode_{rollout_idx:04d}.npz"
                save_eval_episode(
                    episode_path,
                    rollout,
                    initial_observation=initial_observation,
                    initial_goal=initial_goal,
                    initial_info=initial_info,
                    final_info=info,
                    seed=seed,
                )

            result = {
                "rollout": rollout_idx,
                "seed": seed,
                "success": success,
                "final_distance_to_goal": final_distance,
                "steps": steps,
                "episode_path": str(episode_path) if episode_path is not None else None,
            }
            episodes.append(result)
            print(
                f"eval_rollout={rollout_idx:03d} seed={seed} "
                f"success={int(success)} final_distance={final_distance:.4f} steps={steps}"
            )
    finally:
        env.close()

    success_rate = float(np.mean([episode["success"] for episode in episodes])) if episodes else 0.0
    mean_final_distance = (
        float(np.mean([episode["final_distance_to_goal"] for episode in episodes])) if episodes else np.nan
    )
    metrics = {
        "rollouts": args.eval_rollouts,
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "eval_seed_base": args.seed + args.eval_seed_offset,
        "success_rate": success_rate,
        "mean_final_distance": mean_final_distance,
        "rollout_dir": str(rollout_dir) if args.save_eval_rollouts else None,
        "episodes": episodes,
    }
    if args.save_eval_rollouts:
        manifest = {
            "tag": tag,
            "checkpoint": str(ckpt_path),
            "camera_names": camera_names,
            "policy_camera_names": policy.camera_names,
            "eval_width": args.eval_width,
            "eval_height": args.eval_height,
            "metrics": metrics,
        }
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2))

    output_path = args.ckpt_dir / f"eval_{tag}.json"
    output_path.write_text(json.dumps(json_safe(metrics), indent=2))
    print(f"eval_success_rate={success_rate:.3f} mean_final_distance={mean_final_distance:.4f}")
    return metrics


def infer_eval_resolution(dataset_dir: Path, camera_names: list[str]) -> tuple[int, int]:
    first_episode = next(iter(sorted(dataset_dir.glob("episode_*.npz"))), None)
    if first_episode is None:
        return 320, 240
    with np.load(first_episode) as data:
        key = f"{camera_names[0].replace('_camera', '')}_rgb"
        if key not in data:
            key = f"{camera_names[0]}_rgb"
        if key not in data:
            return 320, 240
        height, width = data[key].shape[1:3]
        return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ACT for OpenArm pick-place.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets") / "act_pick_place")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints") / "openarm_act")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-names", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--eval-rollouts", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run simulator eval on policy_best.ckpt every X epochs. 0 disables periodic eval.",
    )
    parser.add_argument("--eval-seed-offset", type=int, default=10000)
    parser.add_argument("--eval-width", type=int, default=None)
    parser.add_argument("--eval-height", type=int, default=None)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument("--save-eval-rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--temporal-agg", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--sticky-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.eval_width is None or args.eval_height is None:
        width, height = infer_eval_resolution(args.dataset_dir, args.camera_names)
        args.eval_width = width if args.eval_width is None else args.eval_width
        args.eval_height = height if args.eval_height is None else args.eval_height
    return args


def main() -> None:
    args = parse_args()
    if args.eval_only:
        ckpt_path = args.ckpt_path or args.ckpt_dir / "policy_best.ckpt"
        evaluate_checkpoint(args, ckpt_path)
        return

    best_ckpt_path, summary = train_policy(args)
    print(
        f"best_checkpoint={best_ckpt_path} "
        f"best_epoch={summary['best_epoch']} best_val_loss={summary['best_val_loss']:.6f}"
    )
    evaluate_checkpoint(args, best_ckpt_path, eval_tag="final_best")


if __name__ == "__main__":
    main()
