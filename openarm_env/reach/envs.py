from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.8,
    "azimuth": 145.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.55]),
}


class OpenArmReachEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        goal_tolerance: float = 0.05,
        terminate_on_success: bool = False,
    ) -> None:
        EzPickle.__init__(
            self,
            render_mode=render_mode,
            frame_skip=frame_skip,
            goal_tolerance=goal_tolerance,
            terminate_on_success=terminate_on_success,
        )

        self.goal_tolerance = goal_tolerance
        self.terminate_on_success = terminate_on_success
        self._model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "openarm_mujoco"
            / "v1"
            / "openarm_reach_scene.xml"
        )
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float64,
        )

        super().__init__(
            model_path=str(self._model_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        self._left_finger_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_finger"
        )
        self._right_finger_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_finger"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]
        self._arm_qpos_slice = slice(0, 7)
        self._finger_qpos_slice = slice(7, 9)
        self._goal = np.zeros(3, dtype=np.float64)

    def _get_ee_position(self) -> np.ndarray:
        left_finger = self.data.xpos[self._left_finger_body_id]
        right_finger = self.data.xpos[self._right_finger_body_id]
        return 0.5 * (left_finger + right_finger)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee_position = self._get_ee_position()
        goal_error = self._goal - ee_position
        return np.concatenate((qpos, qvel, ee_position, self._goal, goal_error))

    def _sample_goal(self) -> np.ndarray:
        return self.np_random.uniform(
            low=np.array([-0.35, -0.30, 0.25], dtype=np.float64),
            high=np.array([0.35, 0.30, 0.85], dtype=np.float64),
        )

    def _set_target_marker(self) -> None:
        self.data.mocap_pos[self._target_mocap_id] = self._goal
        self.data.mocap_quat[self._target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def compute_reward(self, ee_position: np.ndarray, action: np.ndarray) -> float:
        distance = np.linalg.norm(ee_position - self._goal)
        control_penalty = 1e-3 * np.square(action).sum()
        return -(distance + control_penalty)

    def _get_reset_info(self) -> dict[str, float]:
        ee_position = self._get_ee_position()
        distance = np.linalg.norm(ee_position - self._goal)
        return {
            "distance_to_goal": distance,
            "is_success": float(distance < self.goal_tolerance),
        }

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[self._arm_qpos_slice] += self.np_random.uniform(-0.05, 0.05, size=7)
        qpos[self._finger_qpos_slice] = self.np_random.uniform(0.015, 0.03, size=2)
        qvel += self.np_random.uniform(-0.01, 0.01, size=self.model.nv)

        self.set_state(qpos, qvel)
        self._goal = self._sample_goal()
        self._set_target_marker()

        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.do_simulation(action, self.frame_skip)

        ee_position = self._get_ee_position()
        distance = np.linalg.norm(ee_position - self._goal)
        reward = self.compute_reward(ee_position, action)
        terminated = bool(self.terminate_on_success and distance < self.goal_tolerance)
        truncated = False
        info = {
            "distance_to_goal": distance,
            "is_success": float(distance < self.goal_tolerance),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info


class OpenArmBimanualReachEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        goal_tolerance: float = 0.06,
        terminate_on_success: bool = False,
    ) -> None:
        EzPickle.__init__(
            self,
            render_mode=render_mode,
            frame_skip=frame_skip,
            goal_tolerance=goal_tolerance,
            terminate_on_success=terminate_on_success,
        )

        self.goal_tolerance = goal_tolerance
        self.terminate_on_success = terminate_on_success
        self._model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "openarm_mujoco"
            / "v1"
            / "openarm_bimanual_reach_scene.xml"
        )
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(54,),
            dtype=np.float64,
        )

        super().__init__(
            model_path=str(self._model_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        self._left_tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp"
        )
        self._right_tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_hand_tcp"
        )
        self._left_target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_target"
        )
        self._right_target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_target"
        )
        self._left_target_mocap_id = self.model.body_mocapid[self._left_target_body_id]
        self._right_target_mocap_id = self.model.body_mocapid[self._right_target_body_id]
        self._left_arm_qpos_slice = slice(0, 7)
        self._left_finger_qpos_slice = slice(7, 9)
        self._right_arm_qpos_slice = slice(9, 16)
        self._right_finger_qpos_slice = slice(16, 18)
        self._left_goal = np.zeros(3, dtype=np.float64)
        self._right_goal = np.zeros(3, dtype=np.float64)
        self._goal_sampling_center = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self._goal_min_distance_from_center = 0.28

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

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        left_tcp, right_tcp = self._get_tcp_positions()
        left_error = self._left_goal - left_tcp
        right_error = self._right_goal - right_tcp
        return np.concatenate(
            (
                qpos,
                qvel,
                left_tcp,
                right_tcp,
                self._left_goal,
                self._right_goal,
                left_error,
                right_error,
            )
        )

    def _sample_bimanual_goal(self, side: str) -> np.ndarray:
        if side == "left":
            low = np.array([0.12, 0.10, 0.18], dtype=np.float64)
            high = np.array([0.45, 0.42, 0.80], dtype=np.float64)
            fallback = np.array([0.28, 0.24, 0.48], dtype=np.float64)
        elif side == "right":
            low = np.array([0.12, -0.42, 0.18], dtype=np.float64)
            high = np.array([0.45, -0.10, 0.80], dtype=np.float64)
            fallback = np.array([0.28, -0.24, 0.48], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported side: {side!r}")

        for _ in range(128):
            goal = self.np_random.uniform(low=low, high=high)
            if (
                np.linalg.norm(goal - self._goal_sampling_center)
                >= self._goal_min_distance_from_center
            ):
                return goal

        return fallback

    def _sample_left_goal(self) -> np.ndarray:
        return self._sample_bimanual_goal("left")

    def _sample_right_goal(self) -> np.ndarray:
        return self._sample_bimanual_goal("right")

    def _set_target_markers(self) -> None:
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.mocap_pos[self._left_target_mocap_id] = self._left_goal
        self.data.mocap_pos[self._right_target_mocap_id] = self._right_goal
        self.data.mocap_quat[self._left_target_mocap_id] = identity_quat
        self.data.mocap_quat[self._right_target_mocap_id] = identity_quat
        mujoco.mj_forward(self.model, self.data)

    def compute_reward(
        self,
        left_tcp: np.ndarray,
        right_tcp: np.ndarray,
        action: np.ndarray,
    ) -> float:
        left_distance = np.linalg.norm(left_tcp - self._left_goal)
        right_distance = np.linalg.norm(right_tcp - self._right_goal)
        control_penalty = 1e-4 * np.square(action).sum()
        return -(left_distance + right_distance + control_penalty)

    def _get_reset_info(self) -> dict[str, float]:
        left_tcp, right_tcp = self._get_tcp_positions()
        left_distance = np.linalg.norm(left_tcp - self._left_goal)
        right_distance = np.linalg.norm(right_tcp - self._right_goal)
        success = left_distance < self.goal_tolerance and right_distance < self.goal_tolerance
        return {
            "left_distance_to_goal": left_distance,
            "right_distance_to_goal": right_distance,
            "distance_to_goal": left_distance + right_distance,
            "is_success": float(success),
        }

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[self._left_arm_qpos_slice] += self.np_random.uniform(-0.05, 0.05, size=7)
        qpos[self._right_arm_qpos_slice] += self.np_random.uniform(-0.05, 0.05, size=7)
        qpos[self._left_finger_qpos_slice] = self.np_random.uniform(0.015, 0.03, size=2)
        qpos[self._right_finger_qpos_slice] = self.np_random.uniform(0.015, 0.03, size=2)
        qvel += self.np_random.uniform(-0.01, 0.01, size=self.model.nv)

        self.set_state(qpos, qvel)
        self._left_goal = self._sample_left_goal()
        self._right_goal = self._sample_right_goal()
        self._set_target_markers()

        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.do_simulation(action, self.frame_skip)

        left_tcp, right_tcp = self._get_tcp_positions()
        left_distance = np.linalg.norm(left_tcp - self._left_goal)
        right_distance = np.linalg.norm(right_tcp - self._right_goal)
        reward = self.compute_reward(left_tcp, right_tcp, action)
        success = left_distance < self.goal_tolerance and right_distance < self.goal_tolerance
        terminated = bool(self.terminate_on_success and success)
        truncated = False
        info = {
            "left_distance_to_goal": left_distance,
            "right_distance_to_goal": right_distance,
            "distance_to_goal": left_distance + right_distance,
            "is_success": float(success),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info
