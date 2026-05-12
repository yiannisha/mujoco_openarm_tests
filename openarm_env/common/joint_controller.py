from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class JointPositionTargets:
    single_arm_qpos: np.ndarray | None = None
    left_arm_qpos: np.ndarray | None = None
    right_arm_qpos: np.ndarray | None = None
    single_arm_finger_torque: float | None = None
    left_finger_target: float | None = None
    right_finger_target: float | None = None


@dataclass
class OpenArmJointPositionController:
    single_arm_joint_kp: float = 100.0
    single_arm_joint_kd: float = 20.0
    bimanual_joint_kp: float = 120.0
    bimanual_joint_kd: float = 24.0
    default_bimanual_finger_target: float = 0.02
    default_single_arm_finger_torque: float = 0.0

    def act(
        self,
        env,
        targets: JointPositionTargets,
    ) -> np.ndarray:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "_left_arm_qpos_slice") and hasattr(unwrapped, "_right_arm_qpos_slice"):
            return self._act_bimanual(unwrapped, targets)
        if hasattr(unwrapped, "_arm_qpos_slice"):
            return self._act_single_arm(unwrapped, targets)
        raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")

    def _act_single_arm(
        self,
        env,
        targets: JointPositionTargets,
    ) -> np.ndarray:
        if targets.single_arm_qpos is None:
            raise ValueError("single_arm_qpos is required for OpenArmReachEnv control")

        desired_qpos = self._clip_target_qpos(
            np.asarray(targets.single_arm_qpos, dtype=np.float64),
            env.model.jnt_range[env._arm_qpos_slice],
        )
        qacc = np.zeros(env.model.nv, dtype=np.float64)
        qacc[env._arm_qpos_slice] = (
            self.single_arm_joint_kp * (desired_qpos - env.data.qpos[env._arm_qpos_slice])
            - self.single_arm_joint_kd * env.data.qvel[env._arm_qpos_slice]
        )
        tau_full = env.data.qfrc_bias.copy() + self._mass_matrix(env) @ qacc

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[env._arm_qpos_slice] = tau_full[env._arm_qpos_slice].astype(np.float32)
        finger_torque = (
            self.default_single_arm_finger_torque
            if targets.single_arm_finger_torque is None
            else targets.single_arm_finger_torque
        )
        action[7] = finger_torque
        return np.clip(action, env.action_space.low, env.action_space.high)

    def _act_bimanual(
        self,
        env,
        targets: JointPositionTargets,
    ) -> np.ndarray:
        if targets.left_arm_qpos is None or targets.right_arm_qpos is None:
            raise ValueError(
                "left_arm_qpos and right_arm_qpos are required for bimanual OpenArm control"
            )

        left_qpos = self._clip_target_qpos(
            np.asarray(targets.left_arm_qpos, dtype=np.float64),
            env.model.jnt_range[env._left_arm_qpos_slice],
        )
        right_qpos = self._clip_target_qpos(
            np.asarray(targets.right_arm_qpos, dtype=np.float64),
            env.model.jnt_range[env._right_arm_qpos_slice],
        )

        qacc = np.zeros(env.model.nv, dtype=np.float64)
        qacc[env._left_arm_qpos_slice] = (
            self.bimanual_joint_kp * (left_qpos - env.data.qpos[env._left_arm_qpos_slice])
            - self.bimanual_joint_kd * env.data.qvel[env._left_arm_qpos_slice]
        )
        qacc[env._right_arm_qpos_slice] = (
            self.bimanual_joint_kp * (right_qpos - env.data.qpos[env._right_arm_qpos_slice])
            - self.bimanual_joint_kd * env.data.qvel[env._right_arm_qpos_slice]
        )
        tau_full = env.data.qfrc_bias.copy() + self._mass_matrix(env) @ qacc

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[env._left_arm_qpos_slice] = tau_full[env._left_arm_qpos_slice].astype(np.float32)
        action[env._right_arm_qpos_slice] = tau_full[env._right_arm_qpos_slice].astype(np.float32)
        action[env._left_finger_qpos_slice] = (
            self.default_bimanual_finger_target
            if targets.left_finger_target is None
            else targets.left_finger_target
        )
        action[env._right_finger_qpos_slice] = (
            self.default_bimanual_finger_target
            if targets.right_finger_target is None
            else targets.right_finger_target
        )
        return np.clip(action, env.action_space.low, env.action_space.high)

    def _clip_target_qpos(
        self,
        qpos: np.ndarray,
        joint_limits: np.ndarray,
    ) -> np.ndarray:
        return np.clip(qpos, joint_limits[:, 0], joint_limits[:, 1])

    def _mass_matrix(self, env) -> np.ndarray:
        mass = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
        mujoco.mj_fullM(env.model, mass, env.data.qM)
        return mass
