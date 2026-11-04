"""
Orbital-RL: MuJoCo & Gymnasium 3D Yörüngesel Kenetlenme Ortamı
--------------------------------------------------------------
Google DeepMind MuJoCo fizik motorunu Alçak Dünya Yörüngesi (LEO)
3D Clohessy-Wiltshire (CW) bağıl hareket dinamiği ile birleştiren
resmi Gymnasium ortamı.

Mimari:
  - Rijit Cisim Dinamiği & Çarpışmalar: MuJoCo (Multi-Joint Dynamics with Contact)
  - Yörüngesel Kuvvetler (Coriolis & Yerçekimi Gradyanı): qfrc_applied üzerinden enjeksiyon
  - Hedef & Prob Takibi: MuJoCo site_xpos geometrik sensörleri
  - RL Standardı: Gymnasium v1.0+ Env API
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from orbital_config import CONFIG


class OrbitalMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        xml_path=None,
        use_asymptotic_reward=True,
        orbit_altitude=CONFIG.orbit_altitude,
        chaser_mass=CONFIG.chaser_mass,
        max_thrust=CONFIG.max_thrust,
        frame_skip=CONFIG.mujoco_frame_skip,
        max_steps=CONFIG.max_steps,
        docking_radius=CONFIG.docking_radius,
        max_docking_speed=CONFIG.max_docking_speed,
        render_mode=None
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "orbital_docking.xml")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.port_site_id = self.model.site("docking_port_site").id
        self.probe_site_id = self.model.site("chaser_docking_probe").id
        self.chaser_body_id = self.model.body("chaser").id

        self.use_asymptotic_reward = use_asymptotic_reward
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # Astrodinamik parametreleri (Merkezi CONFIG'den)
        self.m = chaser_mass
        self.max_thrust = max_thrust
        self.orbit_altitude = orbit_altitude
        self.docking_radius = docking_radius
        self.max_docking_speed = max_docking_speed

        self.n = CONFIG.mean_motion

        # Eylem Uzayı: 3 Eksen İtici Kuvveti [-1, +1] -> [-15N, +15N]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Gözlem Uzayı: [rx, ry, rz, vx, vy, vz, coriolis_term, distance, speed]
        high_obs = np.array([120.0, 120.0, 120.0, 10.0, 10.0, 10.0, 1.0, 150.0, 15.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, shape=(9,), dtype=np.float32
        )

    def _get_obs(self):
        # Kenetlenme limanı ile prob ucu arasındaki bağıl konum
        port_pos = self.data.site_xpos[self.port_site_id]
        probe_pos = self.data.site_xpos[self.probe_site_id]
        rel_pos = probe_pos - port_pos
        dist = np.linalg.norm(rel_pos)

        vel = self.data.qvel[:3].copy()
        speed = np.linalg.norm(vel)
        coriolis_term = -2.0 * self.n * vel[0]

        return np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],
            vel[0], vel[1], vel[2],
            coriolis_term,
            dist,
            speed
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Başlangıç pozisyonu (Merkezi CONFIG zarfı)
        init_x = np.random.uniform(-CONFIG.init_radial_span, CONFIG.init_radial_span)
        init_y = np.random.uniform(CONFIG.init_along_track_min, CONFIG.init_along_track_max)
        init_z = np.random.uniform(-CONFIG.init_crosstrack_span, CONFIG.init_crosstrack_span)

        # Başlangıç hızı
        init_vx = np.random.uniform(-0.02, 0.02)
        init_vy = -np.random.uniform(CONFIG.init_speed_closing_min, CONFIG.init_speed_closing_max)
        init_vz = np.random.uniform(-0.02, 0.02)

        self.data.qpos[0] = init_x
        self.data.qpos[1] = init_y
        self.data.qpos[2] = init_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        self.data.qvel[0] = init_vx
        self.data.qvel[1] = init_vy
        self.data.qvel[2] = init_vz
        self.data.qvel[3] = 0.0
        self.data.qvel[4] = 0.0
        self.data.qvel[5] = 0.0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "distance": float(obs[7]),
            "speed": float(obs[8]),
            "mean_motion": float(self.n)
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        clipped_action = np.clip(action, -1.0, 1.0)
        thrust_forces = clipped_action * self.max_thrust

        # Sub-stepping (frame_skip döngüsü)
        for _ in range(self.frame_skip):
            # 1. Kontrol İtkisi
            self.data.ctrl[0] = thrust_forces[0]
            self.data.ctrl[1] = thrust_forces[1]
            self.data.ctrl[2] = thrust_forces[2]

            # 2. Clohessy-Wiltshire (CW) Yörüngesel Kuvvetlerini xfrc_applied ile CoM'e Enjekte Et (Sıfır Parazit Tork)
            pos = self.data.qpos[:3]
            vel = self.data.qvel[:3]

            f_cw_x = self.m * (3.0 * (self.n ** 2) * pos[0] + 2.0 * self.n * vel[1])
            f_cw_y = self.m * (-2.0 * self.n * vel[0])
            f_cw_z = self.m * (-(self.n ** 2) * pos[2])

            self.data.xfrc_applied[self.chaser_body_id, :] = 0.0
            self.data.xfrc_applied[self.chaser_body_id, 0] = f_cw_x
            self.data.xfrc_applied[self.chaser_body_id, 1] = f_cw_y
            self.data.xfrc_applied[self.chaser_body_id, 2] = f_cw_z
            self.data.qfrc_applied[:] = 0.0

            # 3. MuJoCo Adımı
            mujoco.mj_step(self.model, self.data)

        # 4. Durum Analizi
        obs = self._get_obs()
        dist = float(obs[7])
        speed = float(obs[8])

        terminated = False
        truncated = False
        docked_success = False
        crashed = False

        if dist <= self.docking_radius:
            if speed <= self.max_docking_speed:
                docked_success = True
                terminated = True
            else:
                crashed = True
                terminated = True

        if dist > 70.0:
            crashed = True
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        # 5. Ödül Hesaplama
        if self.use_asymptotic_reward:
            v_safe = 0.04 + 0.46 * np.tanh(dist / 8.0)
            vel_penalty = -40.0 * max(0.0, speed - v_safe) ** 2
            prox_reward = 20.0 * np.exp(-dist / 4.0)
            vbar_alignment = 5.0 * (obs[1] / (dist + 1e-4)) ** 2
            effort_penalty = -0.05 * float(np.sum(clipped_action ** 2))

            reward = prox_reward + vel_penalty + vbar_alignment + effort_penalty
            if docked_success:
                reward += 200.0
            elif crashed:
                reward -= 100.0
        else:
            reward = -dist - 0.1 * float(np.sum(clipped_action ** 2))
            if docked_success:
                reward += 100.0
            elif crashed:
                reward -= 100.0

        info = {
            "distance": dist,
            "speed": speed,
            "docked_success": docked_success,
            "crashed": crashed,
            "step": self.current_step,
            "thrust": thrust_forces.tolist()
        }

        return obs, float(reward), terminated, truncated, info
