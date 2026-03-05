# Mujoco_robosuite

Minimal Python setup for `mujoco` and `robosuite`, including:
- A MuJoCo double-pendulum demo
- A 2D CartPole swing-up + balance training and deployment pipeline (PPO)

## Project structure

```text
Mujoco_robosuite/
├── .github/
│   └── workflows/
│       └── ci.yml
├── models/
│   ├── cartpole.xml
│   └── double_pendulum.xml
├── scripts/
│   ├── train_deploy_cartpole.py
│   └── run_double_pendulum.py
├── src/
│   └── mujoco_robosuite/
│       ├── __init__.py
│       └── sim/
│           ├── __init__.py
│           ├── cartpole_env.py
│           └── double_pendulum.py
├── tests/
│   └── test_cartpole_env.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10
- Linux desktop with OpenGL support for viewer rendering

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run the double pendulum demo

```bash
source .venv/bin/activate
python scripts/run_double_pendulum.py
```

The script opens a MuJoCo viewer window and simulates a double pendulum under gravity.

## Train and deploy 2D CartPole

Train a PPO policy and then deploy it in a viewer rollout:

```bash
source .venv/bin/activate
python scripts/train_deploy_cartpole.py --mode both --timesteps 100000
```

Train only:

```bash
python scripts/train_deploy_cartpole.py --mode train --timesteps 100000
```

Deploy only (requires saved artifacts):

```bash
python scripts/train_deploy_cartpole.py --mode deploy
```

Useful swing-up options:

```bash
python scripts/train_deploy_cartpole.py \
  --mode train \
  --timesteps 200000 \
  --x-limit 4.8 \
  --reset-mode bottom_biased \
  --bottom-bias-prob 0.8 \
  --bottom-angle-jitter 0.25
```

Environment behavior for CartPole:
- Pole hinge supports continuous 360 degrees (no hinge joint limit).
- Episode endings are time-limit based by default (`--max-episode-steps`).
- Observation order is `[cart_pos, cart_vel, sin_theta, cos_theta, pole_ang_vel]`.

Default artifacts:
- Policy: `artifacts/cartpole_ppo.zip`
- Normalization stats: `artifacts/cartpole_vecnormalize.pkl`

## Train Humanoid

Train only:

```bash
source .venv/bin/activate
python scripts/train_deploy_humanoid.py --mode train --timesteps 1000000
```

Useful throughput options for remote machines:

```bash
python scripts/train_deploy_humanoid.py \
  --mode train \
  --timesteps 1000000 \
  --device auto \
  --n-envs 8 \
  --n-steps 1024 \
  --batch-size 1024 \
  --net-arch 256 256
```

Notes:
- Humanoid training is a mixed CPU/GPU workload: MuJoCo stepping is CPU-bound, policy/value updates run in PyTorch.
- `--mode deploy` uses `render_mode="human"`, so remote headless servers should usually run `--mode train` only.
- `requirements.txt` is intentionally base-only. For GPU training, install a CUDA-enabled PyTorch build that matches the host machine, then verify with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run tests

```bash
source .venv/bin/activate
pytest -q tests/test_cartpole_env.py
```

## CI

GitHub Actions runs:
- `pre-commit` checks on all files
- Headless CartPole environment tests via `pytest`

## Quick environment check

```bash
source .venv/bin/activate
python -c "import mujoco, robosuite, gymnasium, stable_baselines3; print('mujoco', mujoco.__version__); print('robosuite', robosuite.__version__); print('gymnasium', gymnasium.__version__); print('stable_baselines3', stable_baselines3.__version__)"
```

## Troubleshooting

- If the viewer does not open, verify your machine has working OpenGL/GLFW system libraries.
- For headless servers, use offscreen rendering workflows instead of launching the passive viewer.
