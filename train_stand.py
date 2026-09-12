"""
Training PPO per il task "standing" su ANYmal C.

Uso:
    python3 train_stand.py

I checkpoint e i log finiscono in ./logs/ e ./checkpoints/,
leggibili poi con TensorBoard per vedere le curve di training:
    tensorboard --logdir ./logs
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from anymal_env import AnymalStandEnv

XML_PATH = "mujoco_menagerie/anybotics_anymal_c/scene.xml"

TOTAL_TIMESTEPS = 2_000_000   # punto di partenza ragionevole per un task semplice
N_ENVS = 2                     # quanti ambienti in parallelo (CPU-bound, non serve GPU)


def make_env():
    return AnymalStandEnv(xml_path=XML_PATH)


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # N_ENVS ambienti eseguiti in parallelo su processi separati:
    # accelera molto la raccolta di esperienza rispetto a un singolo ambiente
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./logs",
        n_steps=2048,       # step raccolti per ogni env prima di un update
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="anymal_stand",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save("checkpoints/anymal_stand_final")
    print("Training completato, modello salvato in checkpoints/anymal_stand_final")


if __name__ == "__main__":
    main()
