"""
PPO ile Orbital Docking Eğitimi (Stable-Baselines3 Entegrasyonu)
---------------------------------------------------------------
Bu script, 3D Clohessy-Wiltshire ortamında AsymptoticOrbitalReward
ödül fonksiyonunu kullanarak PPO (Proximal Policy Optimization)
ajanını eğitir.

Kullanım:
  python train_ppo.py --timesteps 100000 --save-dir models
"""

import sys
import os

# Scriptin bulunduğu dizini Python modül arama yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np

from orbital_env import OrbitalDockingEnv
from reward_shaping import AsymptoticOrbitalReward, NaiveReward

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def main():
    parser = argparse.ArgumentParser(description="Orbital-RL PPO Training Script")
    parser.add_argument("--timesteps", type=int, default=80000, help="Eğitim toplam adım sayısı")
    parser.add_argument("--naive", action="store_true", help="Hovering üreten NaiveReward ile eğit")
    parser.add_argument("--save-dir", type=str, default="models", help="Model kayıt klasörü")
    args = parser.parse_args()

    if not HAS_SB3:
        print("\n[UYARI] 'stable-baselines3' kütüphanesi henüz yüklü değil!")
        print("Kurulum için: pip install stable-baselines3")
        print("Simülasyon ve ödül testini çalıştırmak için: python simulate_demo.py\n")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    reward_name = "Naive (Hovering Riski)" if args.naive else "Asymptotic & Exponential Shaped"
    print(f"\n==================================================")
    print(f"🚀 ORBITAL-RL PPO EĞİTİMİ BAŞLATILIYOR")
    print(f"Ödül Stratejisi : {reward_name}")
    print(f"Toplam Adım     : {args.timesteps:,}")
    print(f"Fizik Motoru    : 3D Clohessy-Wiltshire (CW) RK4")
    print(f"==================================================\n")

    # 1. Ortamı Oluştur
    reward_shaper = NaiveReward() if args.naive else AsymptoticOrbitalReward()
    raw_env = OrbitalDockingEnv(reward_shaper=reward_shaper)
    env = Monitor(raw_env, filename=os.path.join(log_dir, "train_monitor"))

    try:
        import tensorboard
        tb_log = os.path.join(log_dir, "tb")
    except ImportError:
        tb_log = None

    # 2. PPO Model Konfigürasyonu
    # Uzay aracı itkileri için sürekli eylem uzayına (continuous action space)
    # uygun hiperparametreler:
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
        verbose=1,
        tensorboard_log=tb_log
    )

    # 3. Modeli Eğit
    model.learn(total_timesteps=args.timesteps)

    # 4. Modeli Kaydet
    model_suffix = "naive_hovering" if args.naive else "asymptotic_shaped"
    save_path = os.path.join(args.save_dir, f"ppo_orbital_{model_suffix}")
    model.save(save_path)
    print(f"\n[BAŞARILI] Model kaydedildi: {save_path}.zip")
    print("Test ve görselleştirme için 'evaluate_and_plot.py' çalıştırabilirsiniz.\n")


if __name__ == "__main__":
    main()
