# Mujoco_robosuite

Minimal Python setup for `mujoco` and `robosuite`, including:
- A MuJoCo double-pendulum demo
- A 2D CartPole training + deployment pipeline (PPO)

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

Default artifacts:
- Policy: `artifacts/cartpole_ppo.zip`
- Normalization stats: `artifacts/cartpole_vecnormalize.pkl`

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
