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

    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert info == {}

    env.close()


def test_step_outputs_reward_and_observation_valid() -> None:
    env = CartPoleBalanceEnv(render_mode=None)
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )

    assert obs.shape == (4,)
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


def test_zero_action_eventually_fails_from_pole_angle() -> None:
    env = CartPoleBalanceEnv(render_mode=None)
    env.reset(seed=3)

    done = False
    steps = 0
    last_info = {}
    while not done and steps < 300:
        _, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        done = terminated or truncated
        steps += 1
        last_info = info

    assert done
    assert steps < 300
    assert abs(float(last_info["pole_angle"])) > env.theta_limit

    env.close()
