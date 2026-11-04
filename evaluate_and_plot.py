"""
Model Değerlendirme ve Telemetri Raporlayıcı
--------------------------------------------
Eğitilen PPO modelini (veya test politikasını) 10 farklı başlangıç
noktasından başlatır, başarı oranını (Success Rate), ortalama yakıt
harcamasını (Delta-V) ve çarpışma/hovering istatistiklerini hesaplar.
"""

import sys
import os

# Scriptin bulunduğu dizini Python modül arama yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from orbital_env import OrbitalDockingEnv
from reward_shaping import AsymptoticOrbitalReward, NaiveReward

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def evaluate_policy(model_path=None, num_episodes=10, use_naive=False):
    env = OrbitalDockingEnv(use_naive_reward=use_naive, max_steps=220)
    model = None

    if model_path and os.path.exists(model_path) and HAS_SB3:
        print(f"📦 PPO Modeli yükleniyor: {model_path}")
        model = PPO.load(model_path)
    else:
        print("ℹ️ Model bulunamadı veya henüz eğitilmedi, analitik referans kontrolcüsü çalıştırılıyor...")

    successes = 0
    crashes = 0
    hoverings = 0
    total_fuel_burn = []

    print(f"\n{'='*55}")
    print(f"🛰️  ORBITAL-RL ÇOKLU TEST DEĞERLENDİRMESİ ({num_episodes} Uçuş)")
    print(f"{'='*55}")
    print(f"{'Uçuş':<6} | {'Başlangıç (m)':<15} | {'Son Mesafe':<12} | {'Son Hız':<10} | {'Durum':<12}")
    print(f"{'-'*55}")

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=100 + ep)
        ep_thrust = []
        done = False
        truncated = False

        init_dist = info['distance']

        while not (done or truncated):
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Akıllı referans rehberliği
                pos = env.state[:3]
                vel = env.state[3:6]
                dist = np.linalg.norm(pos)

                if use_naive:
                    if dist > 3.0:
                        target_vel = -pos / (dist + 1e-6) * 0.25
                        action = np.clip((target_vel - vel) * 5.0, -1.0, 1.0)
                    else:
                        action = np.clip(-vel * 8.0, -1.0, 1.0)
                else:
                    target_speed = env.reward_shaper.safe_velocity_limit(dist)
                    desired_vel = -pos / (dist + 1e-6) * target_speed
                    vel_error = desired_vel - vel
                    coriolis_x = -2.0 * env.n * vel[1] * env.m
                    coriolis_y = 2.0 * env.n * vel[0] * env.m
                    force = (vel_error * 80.0) + np.array([coriolis_x, coriolis_y, 0.0])
                    action = np.clip(force / env.max_thrust, -1.0, 1.0)

            ep_thrust.append(np.linalg.norm(action * env.max_thrust))
            obs, reward, done, truncated, info = env.step(action)

        final_dist = info['distance']
        final_speed = info['speed']

        if info['docked_success']:
            status = "🟢 DOCKED"
            successes += 1
        elif info['crashed']:
            status = "💥 CRASH"
            crashes += 1
        else:
            status = "⏸️ HOVER"
            hoverings += 1

        # Yakıt harcaması (Delta-V: Integral of |F| / m * dt)
        delta_v = sum(ep_thrust) / env.m * env.dt
        total_fuel_burn.append(delta_v)

        print(f"#{ep:<5} | {init_dist:6.2f} m        | {final_dist:6.2f} m     | {final_speed:5.3f} m/s | {status}")

    print(f"{'-'*55}")
    print(f"📊 BAŞARI ÖZETİ:")
    print(f"   • Başarılı Kenetlenme: %{(successes / num_episodes) * 100:.1f} ({successes}/{num_episodes})")
    print(f"   • Hovering (Takılma) : %{(hoverings / num_episodes) * 100:.1f} ({hoverings}/{num_episodes})")
    print(f"   • Çarpışma / Kaza   : %{(crashes / num_episodes) * 100:.1f} ({crashes}/{num_episodes})")
    print(f"   • Ort. Delta-V      : {np.mean(total_fuel_burn):.2f} m/s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    trained_model = os.path.join(os.path.dirname(__file__), "models", "ppo_orbital_asymptotic_shaped.zip")
    if not os.path.exists(trained_model):
        trained_model = None
    evaluate_policy(model_path=trained_model, use_naive=False, num_episodes=8)
