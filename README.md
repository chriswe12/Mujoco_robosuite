# Mujoco_robosuite

Minimal Python setup for both `mujoco` and `robosuite`, plus a simple MuJoCo double-pendulum demo.

## Project structure

```text
Mujoco_robosuite/
├── models/
│   └── double_pendulum.xml
├── scripts/
│   └── run_double_pendulum.py
├── src/
│   └── mujoco_robosuite/
│       ├── __init__.py
│       └── sim/
│           ├── __init__.py
│           └── double_pendulum.py
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

## Quick environment check

```bash
source .venv/bin/activate
python -c "import mujoco, robosuite; print('mujoco', mujoco.__version__); print('robosuite', robosuite.__version__)"
```

## Troubleshooting

- If the viewer does not open, verify your machine has working OpenGL/GLFW system libraries.
- For headless servers, use offscreen rendering workflows instead of launching the passive viewer.
