#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mujoco_robosuite.sim.double_pendulum import main


if __name__ == "__main__":
    main()
