#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib

import gymnasium as gym
import mujoco
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_NET_ARCH = [256, 256]
DEFAULT_N_ENVS = max(1, min(8, os.cpu_count() or 1))


def _resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    if path.suffix == ".zip":
        return path
    return path.with_suffix(".zip")


def _default_start_method() -> str:
    available_methods = set(mp.get_all_start_methods())
    if "spawn" in available_methods:
        return "spawn"
    if "forkserver" in available_methods:
        return "forkserver"
    return "spawn"


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


def _make_env_factory(
    render_mode: str | None,
    max_episode_steps: int,
    seed: int,
    index: int,
):
    def _make_env() -> gym.Env:
        env = _make_base_env(render_mode, max_episode_steps)
        env.reset(seed=seed + index)
        env.action_space.seed(seed + index)
        return env

    return _make_env


def _make_vec_env(
    render_mode: str | None,
    max_episode_steps: int,
    seed: int,
    n_envs: int,
    start_method: str,
) -> VecEnv:
    env_fns = [
        _make_env_factory(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            seed=seed,
            index=index,
        )
        for index in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)


def _validate_training_args(args: argparse.Namespace) -> None:
    rollout_batch_size = args.n_envs * args.n_steps
    if args.n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.batch_size > rollout_batch_size:
        raise ValueError(
            f"batch_size={args.batch_size} exceeds rollout batch size "
            f"{rollout_batch_size} (n_envs={args.n_envs} * n_steps={args.n_steps})"
        )


def train(args: argparse.Namespace) -> None:
    _validate_training_args(args)
    vec_env = _make_vec_env(
        render_mode=None,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        n_envs=args.n_envs,
        start_method=args.start_method,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs={
            "net_arch": {
                "pi": list(args.net_arch),
                "vf": list(args.net_arch),
            }
        },
    )

    print(
        "Training config: "
        f"device={model.device}, "
        f"n_envs={args.n_envs}, "
        f"n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, "
        f"rollout_batch_size={args.n_envs * args.n_steps}, "
        f"net_arch={args.net_arch}"
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
        seed=args.seed,
        n_envs=1,
        start_method=args.start_method,
    )
    vec_env = VecNormalize.load(str(args.stats_out), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device=args.device)

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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--net-arch", nargs="+", type=int, default=DEFAULT_NET_ARCH)
    parser.add_argument("--start-method", type=str, default=_default_start_method())
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
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")

    if args.mode in {"train", "both"}:
        train(args)

    if args.mode in {"deploy", "both"}:
        deploy(args)


if __name__ == "__main__":
    main()
