# AGENTS

## Purpose
Contributor and coding-agent guide for this repository.
This file applies to the entire repository tree unless a deeper directory provides a more specific `AGENTS.md`.

## Scope
- Applies to all files and directories in this repo (`.github/`, `models/`, `scripts/`, `src/`, `tests/`, and root config/docs).
- Build and maintain MuJoCo / robosuite examples under `src/` and `scripts/`
- Keep environment behavior reproducible and testable
- Prefer small, reviewable changes

## Development workflow
1. Create/activate virtualenv:
   - `python3.10 -m venv .venv`
   - `source .venv/bin/activate`
2. Install deps:
   - `pip install -r requirements.txt`
3. Run checks before commit:
   - `pre-commit run --all-files`
   - `pytest -q tests/test_cartpole_env.py`

## Project conventions
- Keep simulation code in `src/mujoco_robosuite/sim/`
- Keep runnable entrypoints in `scripts/`
- Keep MuJoCo XML models in `models/`
- Add tests for environment dynamics and API behavior in `tests/`
- Avoid GUI requirements in tests (`render_mode=None` for CI)

## CartPole notes
- `models/cartpole.xml` uses non-colliding decorative geoms (`contype=0`, `conaffinity=0`) to avoid contact constraints blocking cart motion
- Observation order in `CartPoleBalanceEnv`: `[cart_pos, cart_vel, sin_theta, cos_theta, pole_ang_vel]`
- Joint indexing must use MuJoCo address arrays (`jnt_qposadr`, `jnt_dofadr`), not raw joint IDs
