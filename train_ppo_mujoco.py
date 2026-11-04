"""
PPO ile MuJoCo 3D Orbital Docking Eğitimi (Stable-Baselines3 + Gymnasium)
-------------------------------------------------------------------------
Bu script, Google DeepMind MuJoCo fizik motoru ve 3D Clohessy-Wiltshire
yörüngesel dinamiklerini birleştiren OrbitalMujocoEnv ortamında
PPO (Proximal Policy Optimization) ajanını eğitir.

Kullanım:
  python train_ppo_mujoco.py --timesteps 50000
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Orbital-RL PPO Training")
    parser.add_argument("--timesteps", type=int, default=50000, help="Toplam eğitim adım sayısı")
    parser.add_argument("--naive", action="store_true", help="Naive ödül ile eğit")
    parser.add_argument("--save-dir", type=str, default="models", help="Model kayıt klasörü")
    args = parser.parse_args()

    if not HAS_SB3:
        print("[HATA] stable-baselines3 kütüphanesi bulunamadı!")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    reward_type = "Naive (Hovering Riski)" if args.naive else "Asymptotic & Exponential Shaped"
    print("\n" + "=" * 65)
    print("🚀 MUJOCO + GYMNASIUM PPO EĞİTİMİ BAŞLATILIYOR")
    print(f"Fizik Motoru : Google DeepMind MuJoCo (v3.17) + 3D CW Fiziği")
    print(f"Ortam Tipi   : Gymnasium v1.0+")
    print(f"Ödül Modeli  : {reward_type}")
    print(f"Toplam Adım  : {args.timesteps:,}")
    print("=" * 65 + "\n")

    # 1. MuJoCo Gymnasium Ortamını Başlat
    raw_env = OrbitalMujocoEnv(use_asymptotic_reward=not args.naive)
    monitor_path = os.path.join("logs", "mujoco_train_monitor")
    env = Monitor(raw_env, filename=monitor_path)

    # 2. PPO Modeli Konfigürasyonu
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

    # 3. Modeli Eğit
    model.learn(total_timesteps=args.timesteps)

    # 4. Modeli Kaydet
    suffix = "naive" if args.naive else "asymptotic"
    save_path = os.path.join(args.save_dir, f"ppo_mujoco_{suffix}")
    model.save(save_path)
    print(f"\n[BAŞARILI] MuJoCo PPO Modeli kaydedildi: {save_path}.zip\n")


if __name__ == "__main__":
    main()
