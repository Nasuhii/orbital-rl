"""
PPO Eğitimi - Potential-Based Reward Shaping (PBRS) & Anti-Hovering
-------------------------------------------------------------------
Bu script, Ng et al. (1999) potansiyel bazlı ödül şekillendirme teoremini
kullanarak uzay aracı kenetlenmesindeki 'hovering' (yerinde sayma) açığını
matematiksel olarak çözer ve PPO ajanını eğitir.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_env import OrbitalDockingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


class PBRSRewardShaper:
    """
    Potential-Based Reward Shaping (PBRS):
    F(s, a, s') = gamma * Phi(s') - Phi(s)
    Phi(s) = w_phi * exp(-dist / sigma_dist)
    """
    def __init__(self, gamma=0.99, w_phi=250.0, sigma_dist=5.0, dock_bonus=500.0, crash_penalty=300.0):
        self.gamma = gamma
        self.w_phi = w_phi
        self.sigma_dist = sigma_dist
        self.dock_bonus = dock_bonus
        self.crash_penalty = crash_penalty
        self.d_brake = 6.0
        self.v_dock_safe = 0.05
        self.v_approach_max = 0.45

    def safe_velocity_limit(self, dist):
        return self.v_dock_safe + (self.v_approach_max - self.v_dock_safe) * np.tanh(dist / self.d_brake)

    def potential(self, dist):
        return self.w_phi * np.exp(-dist / self.sigma_dist)

    def compute_reward(self, state, action, prev_state, info):
        pos = state[:3]
        vel = state[3:6]
        prev_pos = prev_state[:3] if prev_state is not None else pos

        dist = float(np.linalg.norm(pos))
        prev_dist = float(np.linalg.norm(prev_pos))
        speed = float(np.linalg.norm(vel))
        thrust_norm = float(np.linalg.norm(action))

        # 1. PBRS Şekillendirme: F = gamma * Phi(s') - Phi(s)
        phi_curr = self.potential(dist)
        phi_prev = self.potential(prev_dist)
        r_pbrs = (self.gamma * phi_curr) - phi_prev

        # 2. Hız Sınırı Aşım Cezası
        v_limit = self.safe_velocity_limit(dist)
        excess_speed = max(0.0, speed - v_limit)
        r_vel = - 15.0 * (excess_speed ** 2)

        # 3. İlerleme & Koridor
        closing_speed = (prev_dist - dist)
        r_closing = 5.0 * closing_speed

        # V-bar eksenine (+y) hizalanma
        if dist > 0.1:
            corridor_align = max(0.0, pos[1] / dist)
            r_corridor = 3.0 * (corridor_align ** 2)
        else:
            r_corridor = 3.0

        # Efor ve Zaman maliyeti
        r_effort = - 0.02 * (thrust_norm ** 2) - 0.05

        total = r_pbrs + r_vel + r_closing + r_corridor + r_effort

        if info.get('docked_success', False):
            total += self.dock_bonus
        elif info.get('crashed', False):
            total -= self.crash_penalty

        return float(total)


def train_pbrs(timesteps=30000, save_path="models/ppo_orbital_pbrs_converged.zip"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    shaper = PBRSRewardShaper()
    raw_env = OrbitalDockingEnv(reward_shaper=shaper, max_steps=220)
    env = Monitor(raw_env, filename="logs/pbrs_train_monitor")

    print(f"\n{'='*60}")
    print(f"🚀 PPO PBRS EĞİTİMİ BAŞLATILIYOR ({timesteps:,} Adım)")
    print(f"Ödül Stratejisi : Potential-Based Reward Shaping (PBRS - Anti Hovering)")
    print(f"{'='*60}\n")

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
        verbose=1
    )

    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"\n[BAŞARILI] PBRS PPO Modeli kaydedildi -> {save_path}\n")


if __name__ == "__main__":
    train_pbrs(timesteps=30000)
