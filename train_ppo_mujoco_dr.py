"""
Adım 5: PPO Ajanının MuJoCo Ortamında Domain Randomization ile Eğitimi
--------------------------------------------------------------------
Simülasyon-Gerçeklik (Sim-to-Real) açığını kapatmak için MuJoCo ortamına:
1. Kütle Belirsizliği : 500 kg ± %10 (450 - 550 kg)
2. İtici Dalgalanması : 15 N ± %15 (12.75 - 17.25 N)
3. Sensör Gürültüsü   : Konumda 2 cm RMS, hızda 0.5 cm/s RMS Gauss gürültüsü
4. Rastgele Başlangıç : Along-track 18-25 m, Radial ±2.5 m, Cross-track ±1.8 m
5. Ödül Şekillendirme : Ng et al. Potansiyel Bazlı (PBRS) Anti-Hovering Ödülü
"""

import sys
import os
import argparse
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from orbital_config import CONFIG

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class DomainRandomizedMujocoEnv(gym.Wrapper):
    """Domain Randomization Eklenmiş MuJoCo Gymnasium Sarmalayıcısı"""
    def __init__(self, env):
        super().__init__(env)
        self.nominal_mass = CONFIG.chaser_mass
        self.nominal_thrust = CONFIG.max_thrust

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 1. Kütle Rastgeleleştirme: %10 dalgalanma
        mass_factor = np.random.uniform(0.90, 1.10)
        self.env.m = self.nominal_mass * mass_factor
        
        # 2. İtki Verim Dalgalanması: %15 dalgalanma
        thrust_factor = np.random.uniform(0.85, 1.15)
        self.env.max_thrust = self.nominal_thrust * thrust_factor
        
        # 3. Sensör Gürültüsü Enjeksiyonu
        noisy_obs = self._apply_sensor_noise(obs)
        return noisy_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = self._apply_sensor_noise(obs)
        return noisy_obs, reward, terminated, truncated, info

    def _apply_sensor_noise(self, obs):
        noisy = obs.copy()
        # Konum gürültüsü: sigma = 2 cm (0.02 m)
        noisy[0:3] += np.random.normal(0.0, 0.02, size=3).astype(np.float32)
        # Hız gürültüsü: sigma = 5 mm/s (0.005 m/s)
        noisy[3:6] += np.random.normal(0.0, 0.005, size=3).astype(np.float32)
        return noisy


def train_mujoco_ppo(timesteps=25000, save_path="models/ppo_mujoco_dr_converged.zip"):
    if not HAS_SB3:
        print("[HATA] stable-baselines3 kütüphanesi bulunamadı!")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 ADIM 5: MUJOCO PPO EĞİTİMİ (DOMAIN RANDOMIZATION İLE)")
    print(f"Toplam Timestep       : {timesteps:,}")
    print(f"Kütle Değişimi        : ±%10 (450 - 550 kg)")
    print(f"İtici Verim Değişimi  : ±%15 (12.75 - 17.25 N)")
    print(f"LIDAR/Doppler Gürültü : 2 cm RMS konum / 0.5 cm/s RMS hız")
    print("=" * 70 + "\n")

    raw_env = OrbitalMujocoEnv(use_asymptotic_reward=True, max_steps=CONFIG.max_steps)
    dr_env = DomainRandomizedMujocoEnv(raw_env)
    env = Monitor(dr_env, filename="logs/mujoco_dr_train_monitor")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        device="cpu",
        verbose=1
    )

    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"\n[BAŞARILI] MuJoCo DR Modeli kaydedildi -> {save_path}\n")


if __name__ == "__main__":
    train_mujoco_ppo(timesteps=25000)
