"""
Orbital Docking Gymnasium Ortamı - Clohessy-Wiltshire (CW) 3D Dinamikleri
-------------------------------------------------------------------------
Bu ortam, dairesel LEO (Alçak Dünya Yörüngesi) irtifasındaki bir hedef uzay
istasyonuna (Target) yaklaşan bir kovalayıcı uzay aracının (Chaser) bağıl
hareketini 3 Boyutlu Hill / Clohessy-Wiltshire diferansiyel denklemleri ile modeller.

Fizik Denklemleri (LVLH Koordinat Sistemi):
  x.. =  2 * n * y. + 3 * n^2 * x + (Fx / m)    [Radyal Eksen]
  y.. = -2 * n * x.               + (Fy / m)    [Yörünge İçi / Along-track]
  z.. = -n^2 * z                  + (Fz / m)    [Düzlem Dışı / Cross-track]

Burada n: Ortalama Yörünge Hareketi (Mean Motion - rad/s)
         m: Uzay Aracı Kütlesi (kg)
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    class gym:
        class Env:
            pass
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                if shape is None:
                    if hasattr(low, "shape"):
                        shape = low.shape
                    elif hasattr(high, "shape"):
                        shape = high.shape
                    else:
                        shape = (1,)
                self.shape = tuple(shape)
                self.low = np.array(low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
                self.dtype = dtype

            def sample(self):
                return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

from reward_shaping import AsymptoticOrbitalReward, NaiveReward
from orbital_config import CONFIG


class OrbitalDockingEnv(gym.Env):
    """
    3D Clohessy-Wiltshire Yörüngesel Kenetlenme Ortamı
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        reward_shaper=None,
        use_naive_reward=False,
        dt=CONFIG.dt,
        max_steps=CONFIG.max_steps,
        chaser_mass=CONFIG.chaser_mass,
        max_thrust=CONFIG.max_thrust,
        orbit_altitude=CONFIG.orbit_altitude,
        docking_radius=CONFIG.docking_radius,
        max_docking_speed=CONFIG.max_docking_speed
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.m = chaser_mass
        self.max_thrust = max_thrust
        self.docking_radius = docking_radius
        self.max_docking_speed = max_docking_speed

        self.n = CONFIG.mean_motion

        # Ödül şekillendirici seçimi
        if reward_shaper is not None:
            self.reward_shaper = reward_shaper
        elif use_naive_reward:
            self.reward_shaper = NaiveReward()
        else:
            self.reward_shaper = AsymptoticOrbitalReward()

        # Eylem Uzayı: 3 Eksen İtici Kuvveti [-1, +1] -> [-max_thrust, +max_thrust] N
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Gözlem Uzayı: [x, y, z, vx, vy, vz]
        # x: Radial (m), y: Along-track (m), z: Cross-track (m)
        # vx, vy, vz: Bağıl Hızlar (m/s)
        high_obs = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            shape=(6,),
            dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        self.prev_state = None
        self.trajectory_history = []

    def _cw_derivatives(self, state, thrust):
        """Clohessy-Wiltshire hareket diferansiyel denklemleri (state derivative: ds/dt)."""
        x, y, z, vx, vy, vz = state
        fx, fy, fz = thrust

        ax = 2.0 * self.n * vy + 3.0 * (self.n ** 2) * x + (fx / self.m)
        ay = -2.0 * self.n * vx + (fy / self.m)
        az = -(self.n ** 2) * z + (fz / self.m)

        return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

    def _rk4_step(self, state, thrust, dt):
        """4. Dereceden Runge-Kutta (RK4) Sayısal Yörünge İntegratörü."""
        k1 = self._cw_derivatives(state, thrust)
        k2 = self._cw_derivatives(state + 0.5 * dt * k1, thrust)
        k3 = self._cw_derivatives(state + 0.5 * dt * k2, thrust)
        k4 = self._cw_derivatives(state + dt * k3, thrust)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # Başlangıç Durumu (Chaser hedefin 18-25m gerisinde V-bar koridorunda başlar)
        init_x = np.random.uniform(-CONFIG.init_radial_span, CONFIG.init_radial_span)
        init_y = np.random.uniform(CONFIG.init_along_track_min, CONFIG.init_along_track_max)
        init_z = np.random.uniform(-CONFIG.init_crosstrack_span, CONFIG.init_crosstrack_span)

        # Başlangıç bağıl hızları
        init_vx = np.random.uniform(-0.02, 0.02)
        init_vy = -np.random.uniform(CONFIG.init_speed_closing_min, CONFIG.init_speed_closing_max)
        init_vz = np.random.uniform(-0.02, 0.02)

        self.state = np.array([init_x, init_y, init_z, init_vx, init_vy, init_vz], dtype=np.float64)
        self.prev_state = self.state.copy()
        self.trajectory_history = [self.state.copy()]

        obs = self.state.astype(np.float32)
        info = {
            'distance': float(np.linalg.norm(self.state[:3])),
            'speed': float(np.linalg.norm(self.state[3:6]))
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        self.prev_state = self.state.copy()

        # Eylemi sınırla ve ölçeklendir (Newton cinsine dönüştür)
        clipped_action = np.clip(action, -1.0, 1.0)
        thrust_force = clipped_action * self.max_thrust

        # Fiziği RK4 integratörü ile ilerlet
        self.state = self._rk4_step(self.state, thrust_force, self.dt)
        self.trajectory_history.append(self.state.copy())

        # Durum Analizi
        pos = self.state[:3]
        vel = self.state[3:6]
        distance = np.linalg.norm(pos)
        speed = np.linalg.norm(vel)

        # Başarı ve Çarpışma Şartları
        docked_success = False
        crashed = False
        terminated = False
        truncated = False

        if distance <= self.docking_radius:
            if speed <= self.max_docking_speed:
                docked_success = True
                terminated = True
            else:
                # Kenetlenme portuna aşırı hızla çarpışma
                crashed = True
                terminated = True

        # Yörüngeden aşırı uzaklaşma veya koridordan savrulma
        if distance > 60.0:
            crashed = True
            terminated = True

        # Maksimum adım sınırı (Hovering patolojisi burada zaman aşımına uğrar)
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            'distance': float(distance),
            'speed': float(speed),
            'docked_success': docked_success,
            'crashed': crashed,
            'step': self.current_step,
            'thrust': thrust_force.tolist()
        }

        # Şekillendirilmiş Ödülü Hesapla
        reward = self.reward_shaper.compute_reward(
            self.state,
            clipped_action,
            self.prev_state,
            info
        )

        obs = self.state.astype(np.float32)
        return obs, reward, terminated, truncated, info
