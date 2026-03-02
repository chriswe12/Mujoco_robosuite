from __future__ import annotations

import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mujoco_robosuite.sim.cartpole_env import CartPoleBalanceEnv  # noqa: E402


def test_reset_observation_shape_and_finite() -> None:
    env = CartPoleBalanceEnv(render_mode=None)
    obs, info = env.reset(seed=0)

    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert -1.0 <= float(obs[2]) <= 1.0
    assert -1.0 <= float(obs[3]) <= 1.0
    assert info == {}

    env.close()


def test_step_outputs_reward_and_observation_valid() -> None:
    env = CartPoleBalanceEnv(render_mode=None)
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )

    assert obs.shape == (5,)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cart_pos" in info
    assert "pole_angle" in info
    assert "control" in info

    env.close()


def test_actions_drive_cart_in_opposite_directions() -> None:
    env = CartPoleBalanceEnv(render_mode=None)

    env.reset(seed=2)
    for _ in range(10):
        _, _, term_pos, trunc_pos, info_pos = env.step(
            np.array([1.0], dtype=np.float32)
        )
        if term_pos or trunc_pos:
            break

    env.reset(seed=2)
    for _ in range(10):
        _, _, term_neg, trunc_neg, info_neg = env.step(
            np.array([-1.0], dtype=np.float32)
        )
        if term_neg or trunc_neg:
            break

    x_pos = float(info_pos["cart_pos"])
    x_neg = float(info_neg["cart_pos"])

    assert x_pos > 0.01
    assert x_neg < -0.01

    env.close()


def test_time_limit_only_termination_by_default() -> None:
    env = CartPoleBalanceEnv(render_mode=None, max_episode_steps=25)
    env.reset(seed=3)

    terminated_seen = False
    truncated_seen = False
    steps = 0
    while not truncated_seen and steps < 50:
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        terminated_seen = terminated_seen or terminated
        truncated_seen = truncated
        steps += 1

    assert not terminated_seen
    assert truncated_seen
    assert steps == 25

    env.close()


def test_bottom_reset_mode_starts_near_pi() -> None:
    env = CartPoleBalanceEnv(
        render_mode=None,
        reset_mode="bottom",
        bottom_angle_jitter=0.2,
    )
    env.reset(seed=4)
    angle = float(env.data.qpos[env._hinge_qpos_idx])

    assert abs(angle - np.pi) <= 0.2 + 1e-6

    env.close()


def test_bottom_biased_reset_prefers_bottom_but_not_exclusive() -> None:
    env = CartPoleBalanceEnv(
        render_mode=None,
        reset_mode="bottom_biased",
        bottom_bias_prob=0.8,
        bottom_angle_jitter=0.2,
    )

    near_bottom = 0
    total = 200
    for i in range(total):
        env.reset(seed=i)
        angle = float(env.data.qpos[env._hinge_qpos_idx])
        if abs(angle - np.pi) <= 0.2 + 1e-6:
            near_bottom += 1

    assert near_bottom > 100
    assert near_bottom < total

    env.close()
