from __future__ import annotations

import pathlib
import time
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "cartpole.xml"


class CartPoleBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """2D CartPole swing-up/balance task using MuJoCo dynamics and a Gym API."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: pathlib.Path | str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        terminate_on_limits: bool = False,
        x_limit: float = 4.8,
        reset_mode: str = "bottom_biased",
        bottom_bias_prob: float = 0.8,
        bottom_angle_jitter: float = 0.25,
    ) -> None:
        super().__init__()

        self.model_path = pathlib.Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.terminate_on_limits = bool(terminate_on_limits)
        self.x_limit = float(x_limit)
        self.reset_mode = str(reset_mode)
        self.bottom_bias_prob = float(bottom_bias_prob)
        self.bottom_angle_jitter = float(bottom_angle_jitter)

        if self.render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        if self.reset_mode not in {"bottom_biased", "bottom", "uniform"}:
            raise ValueError(f"Unsupported reset_mode: {self.reset_mode}")
        if not 0.0 <= self.bottom_bias_prob <= 1.0:
            raise ValueError(
                f"bottom_bias_prob must be in [0, 1], got {self.bottom_bias_prob}"
            )
        if self.bottom_angle_jitter < 0.0:
            raise ValueError(
                f"bottom_angle_jitter must be non-negative, got {self.bottom_angle_jitter}"
            )

        slider_jid = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
        )
        hinge_jid = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
        )

        if slider_jid < 0 or hinge_jid < 0:
            raise RuntimeError("Expected joints 'slider' and 'pole_hinge' are missing.")

        self._slider_qpos_idx = int(self.model.jnt_qposadr[slider_jid])
        self._hinge_qpos_idx = int(self.model.jnt_qposadr[hinge_jid])
        self._slider_qvel_idx = int(self.model.jnt_dofadr[slider_jid])
        self._hinge_qvel_idx = int(self.model.jnt_dofadr[hinge_jid])

        self._ctrl_min = float(self.model.actuator_ctrlrange[0, 0])
        self._ctrl_max = float(self.model.actuator_ctrlrange[0, 1])

        self._viewer: mujoco.viewer.Handle | None = None
        self._step_count = 0
        obs_low = np.array([-np.inf, -np.inf, -1.0, -1.0, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _obs(self) -> np.ndarray:
        cart_pos = self.data.qpos[self._slider_qpos_idx]
        pole_angle = self.data.qpos[self._hinge_qpos_idx]
        cart_vel = self.data.qvel[self._slider_qvel_idx]
        pole_ang_vel = self.data.qvel[self._hinge_qvel_idx]
        return np.array(
            [
                cart_pos,
                cart_vel,
                np.sin(pole_angle),
                np.cos(pole_angle),
                pole_ang_vel,
            ],
            dtype=np.float32,
        )

    def _sample_reset_angle(self) -> float:
        if self.reset_mode == "uniform":
            return float(self.np_random.uniform(-np.pi, np.pi))
        if self.reset_mode == "bottom":
            return float(
                self.np_random.uniform(
                    np.pi - self.bottom_angle_jitter, np.pi + self.bottom_angle_jitter
                )
            )
        if self.np_random.random() < self.bottom_bias_prob:
            return float(
                self.np_random.uniform(
                    np.pi - self.bottom_angle_jitter, np.pi + self.bottom_angle_jitter
                )
            )
        return float(self.np_random.uniform(-np.pi, np.pi))

    def _scale_action(self, action: np.ndarray) -> float:
        clipped = float(np.clip(action[0], -1.0, 1.0))
        return (
            self._ctrl_min + (clipped + 1.0) * (self._ctrl_max - self._ctrl_min) / 2.0
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self._slider_qpos_idx] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[self._hinge_qpos_idx] = self._sample_reset_angle()
        self.data.qvel[self._slider_qvel_idx] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[self._hinge_qvel_idx] = self.np_random.uniform(-0.1, 0.1)
        self.data.ctrl[0] = 0.0
        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ctrl = self._scale_action(action)
        self.data.ctrl[0] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        cart_pos = float(self.data.qpos[self._slider_qpos_idx])
        pole_angle = float(self.data.qpos[self._hinge_qpos_idx])

        upright_term = float(np.cos(pole_angle))
        action_penalty = 0.01 * float(action[0] ** 2)
        cart_penalty = 0.09 * float(cart_pos**2)
        reward = upright_term + 0.1 - action_penalty - cart_penalty

        terminated = bool(self.terminate_on_limits and abs(cart_pos) > self.x_limit)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "cart_pos": cart_pos,
            "pole_angle": pole_angle,
            "control": ctrl,
        }

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam"
            )

        if self._viewer.is_running():
            start_time = time.perf_counter()
            self._viewer.sync()
            effective_dt = self.frame_skip * self.model.opt.timestep
            elapsed = time.perf_counter() - start_time
            remaining = effective_dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
