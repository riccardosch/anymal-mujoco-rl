"""
Ambiente Gymnasium custom per ANYmal C in MuJoCo.
Task 1: stare in piedi (standing) senza cadere.

Da qui poi si estende facilmente al task "camminare dritto"
aggiungendo un comando di velocità target nella reward.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class AnymalStandEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Posa "standing" trovata manualmente nello step precedente.
    # Ordine giunti: LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
    #                LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
    STANDING_POSE = np.array([
        0.0,  0.4, -0.8,
        0.0,  0.4, -0.8,
        0.0, -0.4,  0.8,
        0.0, -0.4,  0.8,
    ], dtype=np.float32)

    STANDING_HEIGHT = 0.5  # z della base nella posa standing

    def __init__(self, xml_path, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Quante volte chiamare mj_step per ogni step() dell'ambiente.
        # timestep fisico = 0.002s -> con frame_skip=10 -> control a 50Hz
        self.frame_skip = 10

        # --- Action space ---
        # Offset rispetto alla standing pose, in radianti.
        # +-0.5 rad (~28 gradi) è un range ragionevole per iniziare:
        # abbastanza per camminare, non abbastanza per movimenti assurdi.
        self.action_scale = 0.5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # --- Observation space ---
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        # + joint_pos(12) + joint_vel(12) + last_action(12) = 45
        obs_dim = 3 + 3 + 3 + 12 + 12 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.max_episode_steps = 500  # 500 * 10 * 0.002s = 10s per episodio

    # ------------------------------------------------------------------
    def _get_obs(self):
        # Velocità della base (frame mondo -> non e' esattamente quello che
        # avrebbe un IMU reale, ma per iniziare va bene; lo raffiniamo
        # quando passiamo a rendere il tutto piu' sim-to-real friendly)
        base_lin_vel = self.data.qvel[0:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()

        # Gravita' proiettata nel frame del corpo: dice alla policy
        # quanto e' inclinato il robot rispetto alla verticale.
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        gravity_world = np.array([0, 0, -1.0])
        projected_gravity = rot_mat.T @ gravity_world

        joint_pos = self.data.qpos[7:19].copy()
        joint_vel = self.data.qvel[6:18].copy()

        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity,
            joint_pos, joint_vel, self.last_action,
        ]).astype(np.float32)
        return obs

    def _get_termination(self):
        base_height = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        # componente z dell'asse z del corpo nel frame mondo:
        # 1.0 = perfettamente dritto, <0.6 circa = molto inclinato/caduto
        upright = rot_mat[2, 2]

        fallen = base_height < 0.3 or upright < 0.6
        return fallen

    def _compute_reward(self, action):
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]

        # Per il task "standing": vogliamo velocita' vicine a zero
        # (il robot deve stare fermo e dritto, non cadere ne' vagare)
        lin_vel_penalty = np.sum(np.square(base_lin_vel))
        ang_vel_penalty = np.sum(np.square(base_ang_vel))

        # Penalita' su azioni brusche (motori piu' realistici, meno jitter)
        action_rate_penalty = np.sum(np.square(action - self.last_action))

        # Penalita' se si scosta troppo dall'altezza standing
        height_penalty = (self.data.qpos[2] - self.STANDING_HEIGHT) ** 2

        alive_bonus = 1.0

        reward = (
            alive_bonus
            - 0.05 * lin_vel_penalty
            - 0.05 * ang_vel_penalty
            - 0.01 * action_rate_penalty
            - 0.5 * height_penalty
        )
        return reward

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Posa standing + piccolo rumore casuale sui giunti, pratica
        # standard: rende la policy robusta invece di overfittare
        # su un singolo stato iniziale esatto.
        noise = self.np_random.uniform(-0.05, 0.05, size=12)
        self.data.qpos[7:19] = self.STANDING_POSE + noise
        self.data.qpos[2] = self.STANDING_HEIGHT
        self.data.ctrl[:] = self.STANDING_POSE

        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # L'azione della policy e' un offset scalato rispetto alla standing pose
        target_pos = self.STANDING_POSE + action * self.action_scale
        self.data.ctrl[:] = target_pos

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._get_termination()

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        self.last_action = action.copy()

        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
