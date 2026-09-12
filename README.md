# anymal-mujoco-rl

Training a locomotion policy for the ANYmal C quadruped in MuJoCo using PPO
(Stable-Baselines3) — from standing to walking, CPU-only.

## Status

🚧 Work in progress — Week 1 (environment setup).

## Motivation

No RTX-class GPU available for Isaac Sim / Isaac Lab, so this project uses
MuJoCo (CPU-based) with the official ANYmal C model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Stack

- **MuJoCo** — physics simulation
- **Gymnasium** — standard RL environment interface
- **Stable-Baselines3 (PPO)** — training algorithm

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install mujoco gymnasium stable-baselines3

git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

## Usage

```bash
python3 train_stand.py
```

Monitor training progress:
```bash
tensorboard --logdir ./logs
```

## Project structure

- `anymal_env.py` — custom Gymnasium environment wrapping the MuJoCo model
- `train_stand.py` — PPO training script for the standing task

## What didn't work / lessons learned

_(section to fill in as the project progresses — the honest part)_

## Results

_(videos, training curves — to add)_
