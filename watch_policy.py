"""
Visualizza un checkpoint allenato nel viewer interattivo di MuJoCo.

Uso:
    python3 watch_policy.py checkpoints/anymal_stand_400000_steps.zip

Puoi lanciarlo mentre il training vero gira in background in un altro
terminale (sono processi indipendenti, non si disturbano) per vedere
come migliora la policy nel tempo, guardando checkpoint via via piu' avanzati.
"""

import sys
import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from anymal_env import AnymalStandEnv

XML_PATH = "mujoco_menagerie/anybotics_anymal_c/scene.xml"


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 watch_policy.py <path_checkpoint.zip>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    env = AnymalStandEnv(xml_path=XML_PATH)
    model = PPO.load(checkpoint_path)

    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # deterministic=True: usa l'azione "migliore" secondo la policy,
            # non una campionata casualmente (utile per valutare, non per allenare)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"Episodio finito (terminated={terminated}, truncated={truncated}), reset...")
                obs, info = env.reset()

            viewer.sync()

            # Rallenta per andare a velocita' reale (altrimenti MuJoCo gira
            # molto piu' veloce del tempo reale e sembra "accelerato")
            time_until_next_step = env.model.opt.timestep * env.frame_skip - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
