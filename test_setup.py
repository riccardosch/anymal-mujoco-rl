import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/anybotics_anymal_c/scene.xml")
data = mujoco.MjData(model)

standing_pose = np.array([
    0.0,  0.4, -0.8,
    0.0,  0.4, -0.8,
    0.0, -0.4,  0.8,
    0.0, -0.4,  0.8,
])

data.qpos[7:] = standing_pose
data.qpos[2] = 0.5
data.ctrl[:] = standing_pose   # <-- fondamentale: il target del PD deve combaciare

mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)