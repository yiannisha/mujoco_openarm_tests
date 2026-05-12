from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from openarm_env.common.ik import IterativeIKSolver
from openarm_env.common.joint_controller import JointPositionTargets, OpenArmJointPositionController
from openarm_env.reach.envs import OpenArmBimanualReachEnv, OpenArmReachEnv


@dataclass
class OpenArmIKJointTargetPolicy:
    ik_iterations: int = 40
    ik_step_size: float = 0.7
    ik_damping: float = 0.05
    ik_tolerance: float = 0.002
    bimanual_finger_target: float = 0.02
    single_arm_finger_torque: float = 0.0
    _ik_solver: IterativeIKSolver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_solver = IterativeIKSolver(
            ik_iterations=self.ik_iterations,
            ik_step_size=self.ik_step_size,
            ik_damping=self.ik_damping,
            ik_tolerance=self.ik_tolerance,
        )

    def act(
        self,
        env: OpenArmReachEnv | OpenArmBimanualReachEnv,
    ) -> JointPositionTargets:
        unwrapped = env.unwrapped
        if isinstance(unwrapped, OpenArmBimanualReachEnv):
            left_qpos, right_qpos = self._solve_bimanual_ik(unwrapped)
            return JointPositionTargets(
                left_arm_qpos=left_qpos,
                right_arm_qpos=right_qpos,
                left_finger_target=self.bimanual_finger_target,
                right_finger_target=self.bimanual_finger_target,
            )
        if isinstance(unwrapped, OpenArmReachEnv):
            return JointPositionTargets(
                single_arm_qpos=self._solve_single_arm_ik(unwrapped),
                single_arm_finger_torque=self.single_arm_finger_torque,
            )
        raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")

    def _solve_single_arm_ik(self, env: OpenArmReachEnv) -> np.ndarray:
        qpos = env.data.qpos.copy()
        joint_limits = env.model.jnt_range[env._arm_qpos_slice].copy()

        for _ in range(self.ik_iterations):
            data = self._ik_solver.forward_kinematics(env, qpos)
            ee_position, arm_jacobian = self._single_arm_point_kinematics(env, data)
            position_error = env._goal - ee_position
            if np.linalg.norm(position_error) < self.ik_tolerance:
                break

            delta_q = self._ik_solver.solve_dls_step(arm_jacobian, position_error)
            qpos[env._arm_qpos_slice] += self.ik_step_size * delta_q
            qpos[env._arm_qpos_slice] = np.clip(
                qpos[env._arm_qpos_slice],
                joint_limits[:, 0],
                joint_limits[:, 1],
            )

        return qpos[env._arm_qpos_slice].copy()

    def _solve_bimanual_ik(
        self,
        env: OpenArmBimanualReachEnv,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = env.data.qpos.copy()
        self._ik_solver.solve_arm_to_target(
            env=env,
            qpos=qpos,
            qpos_slice=env._left_arm_qpos_slice,
            body_id=env._left_tcp_body_id,
            goal=env._left_goal,
        )
        self._ik_solver.solve_arm_to_target(
            env=env,
            qpos=qpos,
            qpos_slice=env._right_arm_qpos_slice,
            body_id=env._right_tcp_body_id,
            goal=env._right_goal,
        )
        return (
            qpos[env._left_arm_qpos_slice].copy(),
            qpos[env._right_arm_qpos_slice].copy(),
        )

    def _single_arm_point_kinematics(
        self,
        env: OpenArmReachEnv,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_position = data.xpos[env._left_finger_body_id]
        right_position = data.xpos[env._right_finger_body_id]
        ee_position = 0.5 * (left_position + right_position)

        left_jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
        right_jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(env.model, data, left_jacobian, None, env._left_finger_body_id)
        mujoco.mj_jacBody(env.model, data, right_jacobian, None, env._right_finger_body_id)
        ee_jacobian = 0.5 * (left_jacobian + right_jacobian)
        return ee_position.copy(), ee_jacobian[:, env._arm_qpos_slice]


@dataclass
class OpenArmIKPolicy:
    target_policy: OpenArmIKJointTargetPolicy = field(default_factory=OpenArmIKJointTargetPolicy)
    low_level_controller: OpenArmJointPositionController = field(
        default_factory=OpenArmJointPositionController
    )

    def act(
        self,
        env: OpenArmReachEnv | OpenArmBimanualReachEnv,
    ) -> np.ndarray:
        targets = self.target_policy.act(env)
        return self.low_level_controller.act(env, targets)
