from __future__ import annotations

import pathlib
import time

import mujoco
import mujoco.viewer
import robosuite


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "double_pendulum.xml"


def main() -> None:
    print(f"mujoco version: {mujoco.__version__}")
    print(f"robosuite version: {robosuite.__version__}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Add a small perturbation so motion is obvious on launch.
    data.qpos[0] = 0.45
    data.qpos[1] = -0.25
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")

        while viewer.is_running():
            step_start = time.time()
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
