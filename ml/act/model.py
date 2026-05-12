from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DETR_DIR = REPO_ROOT / "act" / "detr"
if str(ACT_DETR_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DETR_DIR))

from act.detr.models import build_ACT_model  # noqa: E402


OPENARM_QPOS_DIM = 18
OPENARM_ACTION_DIM = 16


@dataclass
class JointPositionTargets:
    single_arm_qpos: np.ndarray | None = None
    left_arm_qpos: np.ndarray | None = None
    right_arm_qpos: np.ndarray | None = None
    single_arm_finger_torque: float | None = None
    left_finger_target: float | None = None
    right_finger_target: float | None = None


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = mu.size(0)
    if batch_size == 0:
        raise ValueError("Cannot compute KL divergence for an empty batch")
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def default_act_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "backbone": "resnet18",
        "pretrained_backbone": False,
        "dilation": False,
        "position_embedding": "sine",
        "camera_names": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
        "enc_layers": 4,
        "dec_layers": 7,
        "dim_feedforward": 3200,
        "hidden_dim": 512,
        "dropout": 0.1,
        "nheads": 8,
        "num_queries": 100,
        "pre_norm": False,
        "masks": False,
        "kl_weight": 10.0,
        "qpos_dim": OPENARM_QPOS_DIM,
        "action_dim": OPENARM_ACTION_DIM,
    }
    config.update(overrides)
    return config


def build_openarm_act_model(config: dict[str, Any]) -> nn.Module:
    return build_ACT_model(SimpleNamespace(**default_act_config(**config)))


def openarm_action_to_targets(action: np.ndarray) -> JointPositionTargets:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != OPENARM_ACTION_DIM:
        raise ValueError(f"Expected {OPENARM_ACTION_DIM} target values, got shape {action.shape}")
    return JointPositionTargets(
        left_arm_qpos=action[0:7].astype(np.float64),
        left_finger_target=float(action[7]),
        right_arm_qpos=action[8:15].astype(np.float64),
        right_finger_target=float(action[15]),
    )


def robot_qpos_from_env(env) -> np.ndarray:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "_right_finger_qpos_slice"):
        raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")
    return unwrapped.data.qpos[: unwrapped._right_finger_qpos_slice.stop].astype(np.float32).copy()


class OpenArmACTPolicy(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        norm_stats: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        temporal_agg: bool = False,
    ) -> None:
        super().__init__()
        self.config = default_act_config(**config)
        self.model = build_openarm_act_model(self.config)
        self.kl_weight = float(self.config["kl_weight"])
        self.chunk_size = int(self.config["num_queries"])
        self.qpos_dim = int(self.config["qpos_dim"])
        self.action_dim = int(self.config["action_dim"])
        self.camera_names = list(self.config["camera_names"])
        self.temporal_agg = temporal_agg
        self.temporal_agg_k = 0.01

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.register_buffer("qpos_mean", torch.zeros(self.qpos_dim))
        self.register_buffer("qpos_std", torch.ones(self.qpos_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        if norm_stats is not None:
            self.set_norm_stats(norm_stats)

        self._cached_actions: np.ndarray | None = None
        self._cached_index = 0
        self._step = 0
        self._temporal_predictions: list[tuple[int, np.ndarray]] = []

        if device is not None:
            self.to(device)

    def configure_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config["lr"])
        lr_backbone = float(self.config["lr_backbone"])
        weight_decay = float(self.config["weight_decay"])
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    def set_norm_stats(self, norm_stats: dict[str, Any]) -> None:
        with torch.no_grad():
            self.qpos_mean.copy_(torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32).view(-1))
            self.qpos_std.copy_(torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32).view(-1))
            self.action_mean.copy_(torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32).view(-1))
            self.action_std.copy_(torch.as_tensor(norm_stats["action_std"], dtype=torch.float32).view(-1))

    def norm_stats(self) -> dict[str, torch.Tensor]:
        return {
            "qpos_mean": self.qpos_mean.detach().cpu(),
            "qpos_std": self.qpos_std.detach().cpu(),
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
        }

    def reset(self) -> None:
        self._cached_actions = None
        self._cached_index = 0
        self._step = 0
        self._temporal_predictions = []

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        image = (image - self.image_mean.to(image.device)) / self.image_std.to(image.device)
        env_state = None
        if actions is None:
            a_hat, _is_pad_hat, _latent = self.model(qpos, image, env_state)
            return a_hat

        if is_pad is None:
            raise ValueError("is_pad is required when actions are provided")
        actions = actions[:, : self.chunk_size]
        is_pad = is_pad[:, : self.chunk_size]
        a_hat, _is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
        total_kld, _dim_wise_kld, _mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * (~is_pad).unsqueeze(-1)).mean()
        kl = total_kld[0]
        loss = l1 + kl * self.kl_weight
        return {"loss": loss, "l1": l1, "kl": kl}

    @torch.inference_mode()
    def predict_action_chunk(self, qpos: np.ndarray, frames: dict[str, np.ndarray]) -> np.ndarray:
        device = next(self.parameters()).device
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device).view(1, -1)
        qpos_tensor = (qpos_tensor - self.qpos_mean) / self.qpos_std
        image_tensor = self._frames_to_tensor(frames, device)
        normalized_actions = self(qpos_tensor, image_tensor)
        action = normalized_actions.squeeze(0).detach().cpu()
        action = action * self.action_std.cpu() + self.action_mean.cpu()
        return action.numpy().astype(np.float32)

    @torch.inference_mode()
    def act(self, env, frames: dict[str, np.ndarray]) -> JointPositionTargets:
        qpos = robot_qpos_from_env(env)
        if self.temporal_agg:
            chunk = self.predict_action_chunk(qpos, frames)
            self._temporal_predictions.append((self._step, chunk))
            candidates = []
            for start_step, prediction in self._temporal_predictions:
                offset = self._step - start_step
                if 0 <= offset < len(prediction):
                    candidates.append(prediction[offset])
            if not candidates:
                raise RuntimeError("Temporal aggregation has no candidate action")
            stacked = np.stack(candidates, axis=0)
            weights = np.exp(-self.temporal_agg_k * np.arange(len(candidates), dtype=np.float32))
            weights = weights / weights.sum()
            action = (stacked * weights[:, None]).sum(axis=0)
        else:
            if self._cached_actions is None or self._cached_index >= len(self._cached_actions):
                self._cached_actions = self.predict_action_chunk(qpos, frames)
                self._cached_index = 0
            action = self._cached_actions[self._cached_index]
            self._cached_index += 1

        self._step += 1
        return openarm_action_to_targets(action)

    def _frames_to_tensor(self, frames: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
        images = []
        for camera_name in self.camera_names:
            if camera_name not in frames:
                raise KeyError(f"Missing camera frame {camera_name!r}")
            image = np.asarray(frames[camera_name])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for {camera_name}, got {image.shape}")
            images.append(image)
        stacked = np.stack(images, axis=0)
        tensor = torch.as_tensor(stacked, dtype=torch.float32, device=device)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        return tensor


def save_openarm_act_checkpoint(
    path: str | Path,
    policy: OpenArmACTPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "norm_stats": policy.norm_stats(),
        "camera_names": policy.camera_names,
        "chunk_size": policy.chunk_size,
        "qpos_dim": policy.qpos_dim,
        "action_dim": policy.action_dim,
        "policy_config": dict(policy.config),
        "temporal_agg": policy.temporal_agg,
    }
    torch.save(checkpoint, path)


def load_openarm_act_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
    temporal_agg: bool | None = None,
) -> tuple[OpenArmACTPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    config = dict(checkpoint.get("policy_config", {}))
    config.setdefault("camera_names", checkpoint.get("camera_names"))
    config.setdefault("num_queries", checkpoint.get("chunk_size", OPENARM_ACTION_DIM))
    config.setdefault("qpos_dim", checkpoint.get("qpos_dim", OPENARM_QPOS_DIM))
    config.setdefault("action_dim", checkpoint.get("action_dim", OPENARM_ACTION_DIM))
    use_temporal_agg = checkpoint.get("temporal_agg", False) if temporal_agg is None else temporal_agg
    policy = OpenArmACTPolicy(
        config=config,
        norm_stats=checkpoint.get("norm_stats"),
        device=device,
        temporal_agg=use_temporal_agg,
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint
