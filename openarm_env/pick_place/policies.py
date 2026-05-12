from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from openarm_env.common.ik import IterativeIKSolver
from openarm_env.common.joint_controller import JointPositionTargets
from openarm_env.pick_place.envs import OpenArmBimanualPickPlaceEnv


@dataclass
class OpenArmEpisodeReplayJointTargetPolicy:
    episode_path: str | Path
    clamp_to_last: bool = True
    _desired_left_arm_qpos: np.ndarray = field(init=False, repr=False)
    _desired_right_arm_qpos: np.ndarray = field(init=False, repr=False)
    _desired_left_finger_target: np.ndarray = field(init=False, repr=False)
    _desired_right_finger_target: np.ndarray = field(init=False, repr=False)
    _episode_length: int = field(init=False, repr=False)
    _step_index: int = field(default=0, init=False, repr=False)
    _last_time: float = field(default=-1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        episode_path = Path(self.episode_path)
        with np.load(episode_path, allow_pickle=False) as episode:
            self._desired_left_arm_qpos = np.asarray(
                episode["desired_left_arm_qpos"], dtype=np.float64
            )
            self._desired_right_arm_qpos = np.asarray(
                episode["desired_right_arm_qpos"], dtype=np.float64
            )
            self._desired_left_finger_target = np.asarray(
                episode["desired_left_finger_target"], dtype=np.float64
            ).reshape(-1)
            self._desired_right_finger_target = np.asarray(
                episode["desired_right_finger_target"], dtype=np.float64
            ).reshape(-1)
            self._episode_length = int(episode["episode_length"][0])

        if self._desired_left_arm_qpos.shape[0] != self._episode_length:
            raise ValueError(
                "Replay episode has inconsistent left-arm target length: "
                f"{self._desired_left_arm_qpos.shape[0]} vs {self._episode_length}"
            )
        if self._desired_right_arm_qpos.shape[0] != self._episode_length:
            raise ValueError(
                "Replay episode has inconsistent right-arm target length: "
                f"{self._desired_right_arm_qpos.shape[0]} vs {self._episode_length}"
            )
        if self._desired_left_finger_target.shape[0] != self._episode_length:
            raise ValueError(
                "Replay episode has inconsistent left-finger target length: "
                f"{self._desired_left_finger_target.shape[0]} vs {self._episode_length}"
            )
        if self._desired_right_finger_target.shape[0] != self._episode_length:
            raise ValueError(
                "Replay episode has inconsistent right-finger target length: "
                f"{self._desired_right_finger_target.shape[0]} vs {self._episode_length}"
            )

    def act(self, env: OpenArmBimanualPickPlaceEnv) -> JointPositionTargets:
        unwrapped = env.unwrapped
        if not isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
            raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")

        if unwrapped.data.time < self._last_time or self._last_time < 0.0:
            self._step_index = 0
        self._last_time = float(unwrapped.data.time)

        step_index = self._step_index
        if step_index >= self._episode_length:
            if not self.clamp_to_last:
                raise IndexError(
                    f"Replay step {step_index} exceeds saved episode length {self._episode_length}"
                )
            step_index = self._episode_length - 1

        targets = JointPositionTargets(
            left_arm_qpos=self._desired_left_arm_qpos[step_index].copy(),
            right_arm_qpos=self._desired_right_arm_qpos[step_index].copy(),
            left_finger_target=float(self._desired_left_finger_target[step_index]),
            right_finger_target=float(self._desired_right_finger_target[step_index]),
        )
        self._step_index += 1
        return targets


@dataclass
class OpenArmBimanualPickPlaceIKJointTargetPolicy:
    ik_iterations: int = 40
    ik_step_size: float = 0.7
    ik_damping: float = 0.05
    ik_tolerance: float = 0.003
    lift_height: float = 0.62
    hover_height: float = 0.62
    contact_height: float = 0.28
    hover_backoff: float = 0.05
    contact_backoff: float = 0.02
    push_overshoot: float = 0.08
    lift_steps: int = 45
    hover_steps: int = 45
    contact_steps: int = 30
    push_steps: int = 45
    contact_finger_target: float = 0.0
    open_finger_target: float = 0.04
    _ik_solver: IterativeIKSolver = field(init=False, repr=False)
    _active_arm: str | None = field(default=None, init=False, repr=False)
    _stage: str = field(default="lift", init=False, repr=False)
    _stage_steps: int = field(default=0, init=False, repr=False)
    _cycle_index: int = field(default=0, init=False, repr=False)
    _last_time: float = field(default=-1.0, init=False, repr=False)
    _hold_left_qpos: np.ndarray | None = field(default=None, init=False, repr=False)
    _hold_right_qpos: np.ndarray | None = field(default=None, init=False, repr=False)
    _lift_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _hover_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _contact_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _push_target: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_solver = IterativeIKSolver(
            ik_iterations=self.ik_iterations,
            ik_step_size=self.ik_step_size,
            ik_damping=self.ik_damping,
            ik_tolerance=self.ik_tolerance,
        )

    def act(self, env: OpenArmBimanualPickPlaceEnv) -> JointPositionTargets:
        unwrapped = env.unwrapped
        if not isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
            raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")

        self._maybe_reset_state(unwrapped)

        if self._stage == "lift":
            target = self._lift_target.copy()
            finger_target = self.open_finger_target
            stage_limit = self.lift_steps
        else:
            if self._stage == "hover" and self._stage_steps == 0:
                self._plan_cycle_targets(unwrapped)

            if self._stage == "hover":
                target = self._hover_target.copy()
                finger_target = self.open_finger_target
                stage_limit = self.hover_steps
            elif self._stage == "contact":
                target = self._contact_target.copy()
                finger_target = self.contact_finger_target
                stage_limit = self.contact_steps
            else:
                target = self._push_target.copy()
                finger_target = self.contact_finger_target
                stage_limit = self.push_steps

        qpos = unwrapped.data.qpos.copy()
        active_qpos_slice = self._active_arm_slice(unwrapped)
        active_body_id = self._active_body_id(unwrapped)
        active_qpos = qpos.copy()
        self._ik_solver.solve_arm_to_target(
            env=unwrapped,
            qpos=active_qpos,
            qpos_slice=active_qpos_slice,
            body_id=active_body_id,
            goal=target,
        )

        self._stage_steps += 1
        if self._stage_steps >= stage_limit:
            self._advance_stage()

        if self._active_arm == "left":
            return JointPositionTargets(
                left_arm_qpos=active_qpos[active_qpos_slice].copy(),
                right_arm_qpos=self._hold_right_qpos.copy(),
                left_finger_target=finger_target,
                right_finger_target=self.open_finger_target,
            )

        return JointPositionTargets(
            left_arm_qpos=self._hold_left_qpos.copy(),
            right_arm_qpos=active_qpos[active_qpos_slice].copy(),
            left_finger_target=self.open_finger_target,
            right_finger_target=finger_target,
        )

    def _maybe_reset_state(self, env: OpenArmBimanualPickPlaceEnv) -> None:
        if env.data.time < self._last_time or self._last_time < 0.0:
            cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = env._get_cube_state()
            self._active_arm = "left" if cube_pos[1] >= 0.0 else "right"
            self._stage = "lift"
            self._stage_steps = 0
            self._cycle_index = 0
            self._hold_left_qpos = env.data.qpos[env._left_arm_qpos_slice].copy()
            self._hold_right_qpos = env.data.qpos[env._right_arm_qpos_slice].copy()
            left_tcp, right_tcp = env._get_tcp_positions()
            active_tcp = left_tcp if self._active_arm == "left" else right_tcp
            self._lift_target = np.array(
                [active_tcp[0], active_tcp[1], self.lift_height],
                dtype=np.float64,
            )
            self._hover_target = None
            self._contact_target = None
            self._push_target = None
        self._last_time = float(env.data.time)

    def _plan_cycle_targets(self, env: OpenArmBimanualPickPlaceEnv) -> None:
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = env._get_cube_state()
        direction_xy = env._goal[:2] - cube_pos[:2]
        distance_xy = float(np.linalg.norm(direction_xy))
        if distance_xy < 1e-6:
            direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction_xy = direction_xy / distance_xy

        self._hover_target = np.array(
            [
                cube_pos[0] - self.hover_backoff * direction_xy[0],
                cube_pos[1] - self.hover_backoff * direction_xy[1],
                self.hover_height,
            ],
            dtype=np.float64,
        )
        self._contact_target = np.array(
            [
                cube_pos[0] - self.contact_backoff * direction_xy[0],
                cube_pos[1] - self.contact_backoff * direction_xy[1],
                self.contact_height,
            ],
            dtype=np.float64,
        )
        self._push_target = np.array(
            [
                env._goal[0] + self.push_overshoot * direction_xy[0],
                env._goal[1] + self.push_overshoot * direction_xy[1],
                self.contact_height,
            ],
            dtype=np.float64,
        )

    def _advance_stage(self) -> None:
        self._stage_steps = 0
        if self._stage == "lift":
            self._stage = "hover"
        elif self._stage == "hover":
            self._stage = "contact"
        elif self._stage == "contact":
            self._stage = "push"
        else:
            self._stage = "hover"
            self._cycle_index += 1

    def _active_arm_slice(self, env: OpenArmBimanualPickPlaceEnv) -> slice:
        if self._active_arm == "left":
            return env._left_arm_qpos_slice
        return env._right_arm_qpos_slice

    def _active_body_id(self, env: OpenArmBimanualPickPlaceEnv) -> int:
        if self._active_arm == "left":
            return env._left_tcp_body_id
        return env._right_tcp_body_id


@dataclass
class OpenArmBimanualPickPlaceGraspIKJointTargetPolicy:
    ik_iterations: int = 80
    ik_step_size: float = 0.7
    ik_damping: float = 0.05
    ik_tolerance: float = 0.003
    lift_height: float = 0.52
    pregrasp_height: float = 0.38
    carry_height: float = 0.42
    grasp_z_offset: float = -0.03
    place_z_offset: float = 0.01
    open_finger_target: float = 0.04
    closed_finger_target: float = 0.002
    lift_steps: int = 30
    pregrasp_steps: int = 40
    grasp_steps: int = 25
    close_steps: int = 35
    carry_steps: int = 40
    transit_steps: int = 50
    place_steps: int = 35
    release_steps: int = 25
    retreat_steps: int = 30
    _ik_solver: IterativeIKSolver = field(init=False, repr=False)
    _active_arm: str | None = field(default=None, init=False, repr=False)
    _stage: str = field(default="lift", init=False, repr=False)
    _stage_steps: int = field(default=0, init=False, repr=False)
    _last_time: float = field(default=-1.0, init=False, repr=False)
    _hold_left_qpos: np.ndarray | None = field(default=None, init=False, repr=False)
    _hold_right_qpos: np.ndarray | None = field(default=None, init=False, repr=False)
    _finger_body_ids: dict[int, dict[str, tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lift_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _pregrasp_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _grasp_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _carry_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _transit_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _place_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _retreat_target: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_solver = IterativeIKSolver(
            ik_iterations=self.ik_iterations,
            ik_step_size=self.ik_step_size,
            ik_damping=self.ik_damping,
            ik_tolerance=self.ik_tolerance,
        )

    def act(self, env: OpenArmBimanualPickPlaceEnv) -> JointPositionTargets:
        unwrapped = env.unwrapped
        if not isinstance(unwrapped, OpenArmBimanualPickPlaceEnv):
            raise TypeError(f"Unsupported env type: {type(unwrapped)!r}")

        self._maybe_reset_state(unwrapped)
        self._maybe_retry_grasp(unwrapped)
        target, finger_target, stage_limit = self._current_stage_command(unwrapped)
        desired_arm_qpos = self._solve_tcp_target(
            unwrapped,
            side=self._active_arm,
            goal=target,
        )

        self._stage_steps += 1
        if self._stage_steps >= stage_limit:
            self._advance_stage()

        if self._active_arm == "left":
            return JointPositionTargets(
                left_arm_qpos=desired_arm_qpos,
                right_arm_qpos=self._hold_right_qpos.copy(),
                left_finger_target=finger_target,
                right_finger_target=self.open_finger_target,
            )

        return JointPositionTargets(
            left_arm_qpos=self._hold_left_qpos.copy(),
            right_arm_qpos=desired_arm_qpos,
            left_finger_target=self.open_finger_target,
            right_finger_target=finger_target,
        )

    def _maybe_reset_state(self, env: OpenArmBimanualPickPlaceEnv) -> None:
        if env.data.time < self._last_time or self._last_time < 0.0:
            self._hold_left_qpos = env.data.qpos[env._left_arm_qpos_slice].copy()
            self._hold_right_qpos = env.data.qpos[env._right_arm_qpos_slice].copy()
            self._active_arm = self._choose_active_arm(env)
            self._stage = "lift"
            self._stage_steps = 0
            self._plan_targets(env)
        self._last_time = float(env.data.time)

    def _maybe_retry_grasp(self, env: OpenArmBimanualPickPlaceEnv) -> None:
        if self._stage not in {"carry", "transit", "place"}:
            return
        if self._stage_steps < 5:
            return
        if self._cube_is_attached(env):
            return

        self._stage = "pregrasp"
        self._stage_steps = 0
        self._plan_targets(env)

    def _choose_active_arm(self, env: OpenArmBimanualPickPlaceEnv) -> str:
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = env._get_cube_state()
        pregrasp_target = np.array(
            [cube_pos[0], cube_pos[1], self.pregrasp_height],
            dtype=np.float64,
        )
        left_error = self._estimate_tcp_error(env, "left", pregrasp_target)
        right_error = self._estimate_tcp_error(env, "right", pregrasp_target)
        return "left" if left_error <= right_error else "right"

    def _estimate_tcp_error(
        self,
        env: OpenArmBimanualPickPlaceEnv,
        side: str,
        goal: np.ndarray,
    ) -> float:
        qpos_slice = env._left_arm_qpos_slice if side == "left" else env._right_arm_qpos_slice
        body_id = env._left_tcp_body_id if side == "left" else env._right_tcp_body_id
        qpos = env.data.qpos.copy()
        joint_limits = env.model.jnt_range[qpos_slice].copy()

        for _ in range(min(self.ik_iterations, 40)):
            data = self._ik_solver.forward_kinematics(env, qpos)
            tcp_position = data.xpos[body_id].copy()
            error = goal - tcp_position
            if np.linalg.norm(error) < self.ik_tolerance:
                break
            jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(env.model, data, jacobian, None, body_id)
            jacobian = jacobian[:, qpos_slice]
            delta_q = self._ik_solver.solve_dls_step(jacobian, error)
            qpos[qpos_slice] += self.ik_step_size * delta_q
            qpos[qpos_slice] = np.clip(
                qpos[qpos_slice],
                joint_limits[:, 0],
                joint_limits[:, 1],
            )

        data = self._ik_solver.forward_kinematics(env, qpos)
        return float(np.linalg.norm(goal - data.xpos[body_id]))

    def _plan_targets(self, env: OpenArmBimanualPickPlaceEnv) -> None:
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = env._get_cube_state()
        left_center, right_center = self._get_gripper_centers(env)
        active_center = left_center if self._active_arm == "left" else right_center

        self._lift_target = np.array(
            [active_center[0], active_center[1], self.lift_height],
            dtype=np.float64,
        )
        self._pregrasp_target = np.array(
            [cube_pos[0], cube_pos[1], self.pregrasp_height],
            dtype=np.float64,
        )
        self._grasp_target = np.array(
            [cube_pos[0], cube_pos[1], cube_pos[2] + self.grasp_z_offset],
            dtype=np.float64,
        )
        self._carry_target = np.array(
            [cube_pos[0], cube_pos[1], self.carry_height],
            dtype=np.float64,
        )
        self._transit_target = np.array(
            [env._goal[0], env._goal[1], self.carry_height],
            dtype=np.float64,
        )
        self._place_target = np.array(
            [env._goal[0], env._goal[1], env._goal[2] + self.place_z_offset],
            dtype=np.float64,
        )
        self._retreat_target = np.array(
            [env._goal[0], env._goal[1], self.lift_height],
            dtype=np.float64,
        )

    def _cube_is_attached(self, env: OpenArmBimanualPickPlaceEnv) -> bool:
        cube_pos, _cube_quat, _cube_linear_velocity, _cube_angular_velocity = env._get_cube_state()
        left_center, right_center = self._get_gripper_centers(env)
        active_center = left_center if self._active_arm == "left" else right_center
        return bool(
            np.linalg.norm(cube_pos - active_center) < 0.10
            and cube_pos[2] > env._cube_goal_height + 0.01
        )

    def _current_stage_command(
        self,
        env: OpenArmBimanualPickPlaceEnv,
    ) -> tuple[np.ndarray, float, int]:
        if self._stage == "lift":
            return self._lift_target.copy(), self.open_finger_target, self.lift_steps
        if self._stage == "pregrasp":
            return self._pregrasp_target.copy(), self.open_finger_target, self.pregrasp_steps
        if self._stage == "grasp":
            return self._grasp_target.copy(), self.open_finger_target, self.grasp_steps
        if self._stage == "close":
            return self._grasp_target.copy(), self.closed_finger_target, self.close_steps
        if self._stage == "carry":
            return self._carry_target.copy(), self.closed_finger_target, self.carry_steps
        if self._stage == "transit":
            return self._transit_target.copy(), self.closed_finger_target, self.transit_steps
        if self._stage == "place":
            return self._place_target.copy(), self.closed_finger_target, self.place_steps
        if self._stage == "release":
            return self._place_target.copy(), self.open_finger_target, self.release_steps
        return self._retreat_target.copy(), self.open_finger_target, self.retreat_steps

    def _advance_stage(self) -> None:
        self._stage_steps = 0
        if self._stage == "lift":
            self._stage = "pregrasp"
        elif self._stage == "pregrasp":
            self._stage = "grasp"
        elif self._stage == "grasp":
            self._stage = "close"
        elif self._stage == "close":
            self._stage = "carry"
        elif self._stage == "carry":
            self._stage = "transit"
        elif self._stage == "transit":
            self._stage = "place"
        elif self._stage == "place":
            self._stage = "release"
        elif self._stage == "release":
            self._stage = "retreat"
        else:
            self._stage = "pregrasp"

    def _solve_tcp_target(
        self,
        env: OpenArmBimanualPickPlaceEnv,
        side: str,
        goal: np.ndarray,
    ) -> np.ndarray:
        qpos_slice = env._left_arm_qpos_slice if side == "left" else env._right_arm_qpos_slice
        body_id = env._left_tcp_body_id if side == "left" else env._right_tcp_body_id
        qpos = env.data.qpos.copy()
        joint_limits = env.model.jnt_range[qpos_slice].copy()

        for _ in range(min(self.ik_iterations, 40)):
            data = self._ik_solver.forward_kinematics(env, qpos)
            tcp_position = data.xpos[body_id].copy()
            error = goal - tcp_position
            if np.linalg.norm(error) < self.ik_tolerance:
                break

            jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(env.model, data, jacobian, None, body_id)
            jacobian = jacobian[:, qpos_slice]
            delta_q = self._ik_solver.solve_dls_step(jacobian, error)
            qpos[qpos_slice] += self.ik_step_size * delta_q
            qpos[qpos_slice] = np.clip(
                qpos[qpos_slice],
                joint_limits[:, 0],
                joint_limits[:, 1],
            )

        return qpos[qpos_slice].copy()

    def _get_gripper_centers(
        self,
        env: OpenArmBimanualPickPlaceEnv,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_center = self._gripper_center_from_data(env, env.data, "left")
        right_center = self._gripper_center_from_data(env, env.data, "right")
        return left_center, right_center

    def _gripper_center_from_data(
        self,
        env: OpenArmBimanualPickPlaceEnv,
        data: mujoco.MjData,
        side: str,
    ) -> np.ndarray:
        finger_ids = self._get_finger_body_ids(env, side)
        return 0.5 * (data.xpos[finger_ids[0]] + data.xpos[finger_ids[1]])

    def _gripper_center_jacobian(
        self,
        env: OpenArmBimanualPickPlaceEnv,
        data: mujoco.MjData,
        side: str,
    ) -> np.ndarray:
        finger_ids = self._get_finger_body_ids(env, side)
        left_jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
        right_jacobian = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(env.model, data, left_jacobian, None, finger_ids[0])
        mujoco.mj_jacBody(env.model, data, right_jacobian, None, finger_ids[1])
        return 0.5 * (left_jacobian + right_jacobian)

    def _get_finger_body_ids(
        self,
        env: OpenArmBimanualPickPlaceEnv,
        side: str,
    ) -> tuple[int, int]:
        model_id = id(env.model)
        cached = self._finger_body_ids.get(model_id)
        if cached is None:
            cached = {
                "left": (
                    mujoco.mj_name2id(
                        env.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "openarm_left_left_finger",
                    ),
                    mujoco.mj_name2id(
                        env.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "openarm_left_right_finger",
                    ),
                ),
                "right": (
                    mujoco.mj_name2id(
                        env.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "openarm_right_left_finger",
                    ),
                    mujoco.mj_name2id(
                        env.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "openarm_right_right_finger",
                    ),
                ),
            }
            self._finger_body_ids[model_id] = cached
        return cached[side]
