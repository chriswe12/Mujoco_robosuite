# Repository Guidelines

## Project Structure & Module Organization
- Core Python package code lives in `src/mujoco_robosuite/`.
- Simulation logic is under `src/mujoco_robosuite/sim/` (for example, `double_pendulum.py`).
- Runnable entry scripts live in `scripts/` (for example, `scripts/run_double_pendulum.py`).
- MuJoCo XML assets live in `models/`.
- Tooling and metadata are in `.pre-commit-config.yaml`, `requirements.txt`, and `.github/workflows/ci.yml`.

## Build, Test, and Development Commands
- `python3.10 -m venv .venv && source .venv/bin/activate`: create and activate the local environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python scripts/run_double_pendulum.py`: run the double pendulum demo viewer.
- `pre-commit run --all-files --show-diff-on-failure`: run all linting/format checks used in CI.
- `python -c "import mujoco, robosuite; print(mujoco.__version__, robosuite.__version__)"`: verify imports and versions.

## Coding Style & Naming Conventions
- Target Python 3.10 with 4-space indentation and type hints where practical.
- Use `snake_case` for modules/functions/variables and `UPPER_SNAKE_CASE` for constants.
- Keep imports sorted with `isort` and code formatted with `black`/`ruff-format`.
- Resolve lint issues with `ruff` before opening a PR.
- Install hooks once per clone: `pre-commit install`.

## Testing Guidelines
- There is no dedicated `tests/` suite yet; current quality gate is pre-commit plus a runtime smoke test.
- For simulation changes, run `pre-commit run --all-files` and `python scripts/run_double_pendulum.py`.
- If you add automated tests, place them in `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style used in history: `feat: ...`, `fix: ...`, `ci: ...`.
- Keep commits focused to one logical change when possible.
- PRs should include a concise summary, linked issue (if applicable), and exact validation commands run.
- Include screenshots or short recordings for viewer/rendering changes.
