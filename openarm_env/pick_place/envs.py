from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

from openarm_env.reach.envs import DEFAULT_CAMERA_CONFIG


PICK_PLACE_INITIAL_ROBOT_POSES = ("basic", "table_ready")
TABLE_READY_LEFT_ARM_QPOS = np.array(
    [-1.41740, -0.54928, 1.57080, 1.15990, 1.55928, -0.61053, -1.40502],
    dtype=np.float64,
)
TABLE_READY_RIGHT_ARM_QPOS = np.array(
    [1.40841, 0.65709, -1.57080, 1.31674, -1.56049, 0.65957, 1.39582],
    dtype=np.float64,
)


class OpenArmBimanualPickPlaceEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        goal_tolerance: float = 0.07,
        terminate_on_success: bool = False,
        sticky_grasp: bool = True,
        sticky_attach_finger_threshold: float = 0.01,
        sticky_release_finger_threshold: float = 0.02,
        sticky_attach_distance: float = 0.05,
        initial_robot_pose: str = "basic",
        width: int = 480,
        height: int = 480,
    ) -> None:
        if initial_robot_pose not in PICK_PLACE_INITIAL_ROBOT_POSES:
            valid = ", ".join(PICK_PLACE_INITIAL_ROBOT_POSES)
            raise ValueError(f"initial_robot_pose must be one of: {valid}")

        EzPickle.__init__(
            self,
            render_mode=render_mode,
            frame_skip=frame_skip,
            goal_tolerance=goal_tolerance,
            terminate_on_success=terminate_on_success,
            sticky_grasp=sticky_grasp,
            sticky_attach_finger_threshold=sticky_attach_finger_threshold,
            sticky_release_finger_threshold=sticky_release_finger_threshold,
            sticky_attach_distance=sticky_attach_distance,
            initial_robot_pose=initial_robot_pose,
            width=width,
            height=height,
        )

        self.goal_tolerance = goal_tolerance
        self.terminate_on_success = terminate_on_success
        self.sticky_grasp = sticky_grasp
        self._sticky_attach_finger_threshold = sticky_attach_finger_threshold
        self._sticky_release_finger_threshold = sticky_release_finger_threshold
        self._sticky_attach_distance = sticky_attach_distance
        self.initial_robot_pose = initial_robot_pose
        self._model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "openarm_mujoco"
            / "v1"
            / "openarm_bimanual_pick_place_scene.xml"
        )
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(74,),
            dtype=np.float64,
        )

        super().__init__(
            model_path=str(self._model_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=width,
            height=height,
        )

        self._left_tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp"
        )
        self._right_tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_hand_tcp"
        )
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "task_cube"
        )
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "task_cube_joint"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target"
        )
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]
        self._left_arm_qpos_slice = slice(0, 7)
        self._left_finger_qpos_slice = slice(7, 9)
        self._right_arm_qpos_slice = slice(9, 16)
        self._right_finger_qpos_slice = slice(16, 18)

        cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]
        cube_qvel_addr = self.model.jnt_dofadr[self._cube_joint_id]
        self._cube_qpos_slice = slice(cube_qpos_addr, cube_qpos_addr + 7)
        self._cube_qvel_slice = slice(cube_qvel_addr, cube_qvel_addr + 6)

        self._table_center = np.array([0.32, 0.0], dtype=np.float64)
        self._table_half_extents = np.array([0.18, 0.28], dtype=np.float64)
        self._table_height = 0.28
        self._cube_half_size = 0.02
        self._cube_goal_height = self._table_height + self._cube_half_size
        self._sample_margin = np.array([0.04, 0.05], dtype=np.float64)
        self._cube_depth_fraction = 5.0 / 6.0
        self._goal_region_scale = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        self._min_cube_goal_distance = 0.12
        self._goal_speed_tolerance = 0.15
        self._goal_height_tolerance = 0.025
        self._goal = np.array(
            [self._table_center[0], self._table_center[1], self._cube_goal_height],
            dtype=np.float64,
        )
        self._attached_side: str | None = None
        self._attached_local_pos = np.zeros(3, dtype=np.float64)
        self._attached_local_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_action_space(self):
        low = np.zeros(self.model.nu, dtype=np.float32)
        high = np.zeros(self.model.nu, dtype=np.float32)

        ctrlrange = self.model.actuator_ctrlrange.astype(np.float32)
        forcerange = self.model.actuator_forcerange.astype(np.float32)
        has_ctrlrange = np.abs(ctrlrange).sum(axis=1) > 0

        low[has_ctrlrange] = ctrlrange[has_ctrlrange, 0]
        high[has_ctrlrange] = ctrlrange[has_ctrlrange, 1]
        low[~has_ctrlrange] = forcerange[~has_ctrlrange, 0]
        high[~has_ctrlrange] = forcerange[~has_ctrlrange, 1]

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _get_tcp_positions(self) -> tuple[np.ndarray, np.ndarray]:
        left_tcp = self.data.xpos[self._left_tcp_body_id].copy()
        right_tcp = self.data.xpos[self._right_tcp_body_id].copy()
        return left_tcp, right_tcp

    def _get_cube_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cube_qpos = self.data.qpos[self._cube_qpos_slice].copy()
        cube_pos = cube_qpos[:3]
        cube_quat = cube_qpos[3:7]
        cube_spatial_velocity = self.data.cvel[self._cube_body_id].copy()
        cube_angular_velocity = cube_spatial_velocity[:3]
        cube_linear_velocity = cube_spatial_velocity[3:]
        return cube_pos, cube_quat, cube_linear_velocity, cube_angular_velocity

    def _clear_attachment(self) -> None:
        self._attached_side = None
        self._attached_local_pos.fill(0.0)
        self._attached_local_quat[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _finger_command(self, action: np.ndarray, side: str) -> float:
        finger_slice = (
            self._left_finger_qpos_slice if side == "left" else self._right_finger_qpos_slice
        )
        return float(np.mean(action[finger_slice]))

    def _tcp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body_id = self._left_tcp_body_id if side == "left" else self._right_tcp_body_id
        tcp_pos = self.data.xpos[body_id].copy()
        tcp_quat = self.data.xquat[body_id].copy()
        tcp_mat = self.data.xmat[body_id].reshape(3, 3).copy()
        return tcp_pos, tcp_quat, tcp_mat

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    def _quat_multiply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.array(
            [
                lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
                lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
                lhs[0] * rhs[2] - lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
                lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0],
            ],
            dtype=np.float64,
        )

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def _attach_cube_to_side(self, side: str) -> None:
        cube_pos, cube_quat, _cube_linear_velocity, _cube_angular_velocity = self._get_cube_state()
        tcp_pos, tcp_quat, tcp_mat = self._tcp_pose(side)
        self._attached_side = side
        self._attached_local_pos = tcp_mat.T @ (cube_pos - tcp_pos)
        self._attached_local_quat = self._normalize_quat(
            self._quat_multiply(self._quat_conjugate(tcp_quat), cube_quat)
        )

    def _maybe_attach_cube(self, action: np.ndarray) -> None:
        if not self.sticky_grasp or self._attached_side is not None:
            return

        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = self._get_cube_state()
        candidates: list[tuple[float, str]] = []
        for side in ("left", "right"):
            if self._finger_command(action, side) > self._sticky_attach_finger_threshold:
                continue
            tcp_pos, _tcp_quat, _tcp_mat = self._tcp_pose(side)
            distance = float(np.linalg.norm(cube_pos - tcp_pos))
            if distance <= self._sticky_attach_distance:
                candidates.append((distance, side))

        if not candidates:
            return

        candidates.sort(key=lambda item: item[0])
        self._attach_cube_to_side(candidates[0][1])

    def _maybe_release_cube(self, action: np.ndarray) -> None:
        if not self.sticky_grasp or self._attached_side is None:
            return
        if self._finger_command(action, self._attached_side) > self._sticky_release_finger_threshold:
            self._clear_attachment()

    def _apply_attached_cube_pose(self) -> None:
        if not self.sticky_grasp or self._attached_side is None:
            return

        tcp_pos, tcp_quat, tcp_mat = self._tcp_pose(self._attached_side)
        cube_pos = tcp_pos + tcp_mat @ self._attached_local_pos
        cube_quat = self._normalize_quat(
            self._quat_multiply(tcp_quat, self._attached_local_quat)
        )
        self.data.qpos[self._cube_qpos_slice] = np.concatenate((cube_pos, cube_quat))
        self.data.qvel[self._cube_qvel_slice] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        left_tcp, right_tcp = self._get_tcp_positions()
        cube_pos, cube_quat, cube_linear_velocity, cube_angular_velocity = (
            self._get_cube_state()
        )
        cube_error = self._goal - cube_pos
        return np.concatenate(
            (
                qpos,
                qvel,
                left_tcp,
                right_tcp,
                cube_pos,
                cube_quat,
                cube_linear_velocity,
                cube_angular_velocity,
                self._goal,
                cube_error,
            )
        )

    def _sample_table_xy(self) -> np.ndarray:
        low = self._table_center - self._table_half_extents + self._sample_margin
        high = self._table_center + self._table_half_extents - self._sample_margin
        usable_depth = (high[0] - low[0]) * self._cube_depth_fraction
        high[0] = low[0] + usable_depth
        return self.np_random.uniform(low=low, high=high)

    def _sample_goal_xy(self) -> np.ndarray:
        goal_half_extents = self._table_half_extents * self._goal_region_scale
        low = self._table_center - goal_half_extents
        high = self._table_center + goal_half_extents
        return self.np_random.uniform(low=low, high=high)

    def _sample_cube_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(128):
            cube_xy = self._sample_table_xy()
            goal_xy = self._sample_goal_xy()
            if np.linalg.norm(cube_xy - goal_xy) >= self._min_cube_goal_distance:
                cube_pos = np.array(
                    [cube_xy[0], cube_xy[1], self._cube_goal_height], dtype=np.float64
                )
                goal_pos = np.array(
                    [goal_xy[0], goal_xy[1], self._cube_goal_height], dtype=np.float64
                )
                return cube_pos, goal_pos

        cube_xy = self._table_center + np.array([-0.06, 0.12], dtype=np.float64)
        goal_xy = self._table_center + np.array([0.04, -0.06], dtype=np.float64)
        cube_pos = np.array([cube_xy[0], cube_xy[1], self._cube_goal_height], dtype=np.float64)
        goal_pos = np.array([goal_xy[0], goal_xy[1], self._cube_goal_height], dtype=np.float64)
        return cube_pos, goal_pos

    def _set_target_marker(self) -> None:
        marker_pos = self._goal.copy()
        marker_pos[2] = self._table_height + 0.001
        self.data.mocap_pos[self._target_mocap_id] = marker_pos
        self.data.mocap_quat[self._target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def _cube_goal_metrics(self) -> tuple[float, float, float, bool]:
        cube_pos, _cube_quat, cube_linear_velocity, _cube_angular_velocity = (
            self._get_cube_state()
        )
        xy_distance = np.linalg.norm(cube_pos[:2] - self._goal[:2])
        distance = np.linalg.norm(cube_pos - self._goal)
        cube_speed = np.linalg.norm(cube_linear_velocity)
        height_error = abs(cube_pos[2] - self._cube_goal_height)
        success = (
            xy_distance < self.goal_tolerance
            and height_error < self._goal_height_tolerance
            and cube_speed < self._goal_speed_tolerance
        )
        return xy_distance, distance, cube_speed, bool(success)

    def compute_reward(self, success: bool) -> float:
        return float(success)

    def _get_reset_info(self) -> dict[str, float]:
        xy_distance, distance, cube_speed, success = self._cube_goal_metrics()
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = (
            self._get_cube_state()
        )
        return {
            "xy_distance_to_goal": xy_distance,
            "distance_to_goal": distance,
            "cube_height": cube_pos[2],
            "cube_speed": cube_speed,
            "sticky_attached": float(self._attached_side is not None),
            "is_success": float(success),
        }

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self._clear_attachment()

        if self.initial_robot_pose == "table_ready":
            qpos[self._left_arm_qpos_slice] = TABLE_READY_LEFT_ARM_QPOS
            qpos[self._right_arm_qpos_slice] = TABLE_READY_RIGHT_ARM_QPOS

        qpos[self._left_arm_qpos_slice] += self.np_random.uniform(-0.05, 0.05, size=7)
        qpos[self._right_arm_qpos_slice] += self.np_random.uniform(-0.05, 0.05, size=7)
        qpos[self._left_arm_qpos_slice] = self._clip_target_qpos(
            qpos[self._left_arm_qpos_slice],
            self.model.jnt_range[self._left_arm_qpos_slice],
        )
        qpos[self._right_arm_qpos_slice] = self._clip_target_qpos(
            qpos[self._right_arm_qpos_slice],
            self.model.jnt_range[self._right_arm_qpos_slice],
        )
        qpos[self._left_finger_qpos_slice] = self.np_random.uniform(0.015, 0.03, size=2)
        qpos[self._right_finger_qpos_slice] = self.np_random.uniform(0.015, 0.03, size=2)
        qvel += self.np_random.uniform(-0.01, 0.01, size=self.model.nv)

        cube_pos, self._goal = self._sample_cube_and_goal()
        cube_yaw = self.np_random.uniform(-np.pi, np.pi)
        cube_quat = np.array(
            [np.cos(0.5 * cube_yaw), 0.0, 0.0, np.sin(0.5 * cube_yaw)],
            dtype=np.float64,
        )
        qpos[self._cube_qpos_slice] = np.concatenate((cube_pos, cube_quat))
        qvel[self._cube_qvel_slice] = 0.0

        self.set_state(qpos, qvel)
        self._set_target_marker()

        return self._get_obs()

    def _clip_target_qpos(
        self,
        qpos: np.ndarray,
        joint_limits: np.ndarray,
    ) -> np.ndarray:
        return np.clip(qpos, joint_limits[:, 0], joint_limits[:, 1])

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._maybe_release_cube(action)
        self._apply_attached_cube_pose()
        self.do_simulation(action, self.frame_skip)
        if self._attached_side is None:
            self._maybe_attach_cube(action)
        self._maybe_release_cube(action)
        self._apply_attached_cube_pose()

        xy_distance, distance, cube_speed, success = self._cube_goal_metrics()
        reward = self.compute_reward(success)
        terminated = bool(self.terminate_on_success and success)
        truncated = False
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = (
            self._get_cube_state()
        )
        info = {
            "xy_distance_to_goal": xy_distance,
            "distance_to_goal": distance,
            "cube_height": cube_pos[2],
            "cube_speed": cube_speed,
            "sticky_attached": float(self._attached_side is not None),
            "sticky_side": self._attached_side or "",
            "is_success": float(success),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info
