from anymal_env import AnymalStandEnv
import numpy as np

env = AnymalStandEnv(xml_path="mujoco_menagerie/anybotics_anymal_c/scene.xml")
obs, info = env.reset(seed=42)
print("Obs shape:", obs.shape)

total_reward = 0
for i in range(500):
    action = np.zeros(12, dtype=np.float32)  # azioni zero = mantieni standing
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Finito allo step {i}")
        break

print(f"Reward totale: {total_reward:.2f}")