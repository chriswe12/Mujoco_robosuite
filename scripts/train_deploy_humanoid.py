#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

import gymnasium as gym
import mujoco
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    if path.suffix == ".zip":
        return path
    return path.with_suffix(".zip")


def _make_base_env(
    render_mode: str | None,
    max_episode_steps: int,
) -> gym.Env:
    env = gym.make(
        "Humanoid-v5",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    return Monitor(env)


def _make_vec_env(
    render_mode: str | None,
    max_episode_steps: int,
) -> DummyVecEnv:
    return DummyVecEnv([lambda: _make_base_env(render_mode, max_episode_steps)])


def train(args: argparse.Namespace) -> None:
    vec_env = _make_vec_env(render_mode=None, max_episode_steps=args.max_episode_steps)
    vec_env.seed(args.seed)
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
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)

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

    vec_env = _make_vec_env(
        render_mode="human",
        max_episode_steps=args.max_episode_steps,
    )
    vec_env = VecNormalize.load(str(args.stats_out), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env)

    for episode in range(1, args.deploy_episodes + 1):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = vec_env.step(action)
            total_reward += float(rewards[0])
            steps += 1
            done = bool(dones[0])

        print(f"Episode {episode}: steps={steps} total_reward={total_reward:.3f}")

    vec_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and deploy a MuJoCo Humanoid forward-walking policy."
    )
    parser.add_argument("--mode", choices=["train", "deploy", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deploy-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--model-out",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts" / "humanoid_ppo.zip",
    )
    parser.add_argument(
        "--stats-out",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts" / "humanoid_vecnormalize.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"mujoco version: {mujoco.__version__}")
    print(f"gymnasium version: {gym.__version__}")
    print(f"stable_baselines3 version: {stable_baselines3.__version__}")

    if args.mode in {"train", "both"}:
        train(args)

    if args.mode in {"deploy", "both"}:
        deploy(args)


if __name__ == "__main__":
    main()
