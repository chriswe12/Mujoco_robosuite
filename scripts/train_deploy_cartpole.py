#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import robosuite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mujoco_robosuite.sim.cartpole_env import CartPoleBalanceEnv  # noqa: E402


def _resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    if path.suffix == ".zip":
        return path
    return path.with_suffix(".zip")


def make_env(render_mode: str | None = None, max_episode_steps: int = 1000):
    def _factory() -> CartPoleBalanceEnv:
        return CartPoleBalanceEnv(
            render_mode=render_mode, max_episode_steps=max_episode_steps
        )

    return _factory


def train(args: argparse.Namespace) -> None:
    vec_env = DummyVecEnv(
        [make_env(render_mode=None, max_episode_steps=args.max_episode_steps)]
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)

    model_path = _resolve_model_path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    vec_env.save(str(args.stats_out))
    vec_env.close()

    print(f"Saved policy to: {model_path}")
    print(f"Saved normalization stats to: {args.stats_out}")


def deploy(args: argparse.Namespace) -> None:
    model_path = _resolve_model_path(args.model_out)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not args.stats_out.exists():
        raise FileNotFoundError(f"Missing normalization artifact: {args.stats_out}")

    vec_env = DummyVecEnv(
        [make_env(render_mode="human", max_episode_steps=args.max_episode_steps)]
    )
    vec_env = VecNormalize.load(str(args.stats_out), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env)

    for episode in range(1, args.deploy_episodes + 1):
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        angle_accum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += float(rewards[0])
            episode_steps += 1

            info = infos[0]
            angle_accum += abs(float(info.get("pole_angle", 0.0)))

            done = bool(dones[0])

        mean_abs_angle = angle_accum / max(episode_steps, 1)
        print(
            f"Episode {episode}: steps={episode_steps} total_reward={episode_reward:.3f} "
            f"mean_abs_pole_angle={mean_abs_angle:.4f}"
        )

    vec_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and deploy a 2D CartPole balancer."
    )
    parser.add_argument("--mode", choices=["train", "deploy", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--deploy-episodes", type=int, default=3)
    parser.add_argument(
        "--model-out",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts" / "cartpole_ppo.zip",
    )
    parser.add_argument(
        "--stats-out",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts" / "cartpole_vecnormalize.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"mujoco version: {mujoco.__version__}")
    print(f"robosuite version: {robosuite.__version__}")

    if args.mode in {"train", "both"}:
        train(args)

    if args.mode in {"deploy", "both"}:
        deploy(args)


if __name__ == "__main__":
    main()
