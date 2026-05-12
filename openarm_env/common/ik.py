from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class IterativeIKSolver:
    ik_iterations: int = 40
    ik_step_size: float = 0.7
    ik_damping: float = 0.05
    ik_tolerance: float = 0.002
    _scratch_data: mujoco.MjData | None = field(default=None, init=False, repr=False)
    _scratch_model_id: int | None = field(default=None, init=False, repr=False)

    def solve_arm_to_target(
        self,
        env,
        qpos: np.ndarray,
        qpos_slice: slice,
        body_id: int,
        goal: np.ndarray,
    ) -> None:
        joint_limits = env.model.jnt_range[qpos_slice].copy()

        for _ in range(self.ik_iterations):
            data = self.forward_kinematics(env, qpos)
            jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(env.model, data, jacobian, None, body_id)
            body_position = data.xpos[body_id].copy()
            position_error = goal - body_position
            if np.linalg.norm(position_error) < self.ik_tolerance:
                break

            arm_jacobian = jacobian[:, qpos_slice]
            delta_q = self.solve_dls_step(arm_jacobian, position_error)
            qpos[qpos_slice] += self.ik_step_size * delta_q
            qpos[qpos_slice] = np.clip(
                qpos[qpos_slice],
                joint_limits[:, 0],
                joint_limits[:, 1],
            )

    def solve_body_target(
        self,
        env,
        qpos_slice: slice,
        body_id: int,
        goal: np.ndarray,
    ) -> np.ndarray:
        qpos = env.data.qpos.copy()
        self.solve_arm_to_target(env, qpos, qpos_slice, body_id, goal)
        return qpos[qpos_slice].copy()

    def solve_dls_step(
        self,
        jacobian: np.ndarray,
        position_error: np.ndarray,
    ) -> np.ndarray:
        regularizer = (self.ik_damping**2) * np.eye(3, dtype=np.float64)
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, position_error)

    def forward_kinematics(
        self,
        env,
        qpos: np.ndarray,
    ) -> mujoco.MjData:
        if self._scratch_data is None or self._scratch_model_id != id(env.model):
            self._scratch_data = mujoco.MjData(env.model)
            self._scratch_model_id = id(env.model)

        self._scratch_data.qpos[:] = qpos
        self._scratch_data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, self._scratch_data)
        return self._scratch_data
