"""
Orbital-RL: Champion-PPO Terminal Yaklaşma & PWPF+CBF %100 İnce Ayar (Fine-Tuning)
---------------------------------------------------------------------------------
Mevcut Champion-PPO modelini yükler, PWPF=%100 ve CBF=%100 açıkken terminal
yaklaşma ağırlıklı müfredat ve güçlendirilmiş hover/stagnation cezası ile
ince ayar yapar, ardından 50 bölümlük saf MuJoCo doğrulaması çalıştırır.
"""

import os
import sys
import json
import time
import shutil
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mujoco_champion_ppo import (
    build_vec_env,
    evaluate_pure_mujoco,
    linear_schedule,
    CHAMPION_MODEL_PATH,
    STATS_OUTPUT_PATH,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BACKUP_MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion_v3_backup.zip")
CANDIDATE_MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion_v4.zip")


def run_terminal_finetune(finetune_steps=700_000, n_envs=8):
    print("=" * 80)
    print("ORBITAL-RL // CHAMPION-PPO TERMINAL FINE-TUNING (PWPF=100%, CBF=100%)")
    print("=" * 80)

    if not os.path.exists(CHAMPION_MODEL_PATH):
        raise FileNotFoundError(f"Champion modeli bulunamadı: {CHAMPION_MODEL_PATH}")

    if not os.path.exists(BACKUP_MODEL_PATH):
        shutil.copy2(CHAMPION_MODEL_PATH, BACKUP_MODEL_PATH)
        print(f"📦 Yedek alındı: {BACKUP_MODEL_PATH}")

    # Eski başarı oranını oku
    prev_success = 0.0
    if os.path.exists(STATS_OUTPUT_PATH):
        with open(STATS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            prev_stats = json.load(f)
            prev_success = float(prev_stats.get("success_rate", 0.0))
    print(f"📊 Mevcut Baseline (pure_cbf_pwpf): %{prev_success:.1f}")

    # Terminal ağırlıklı ama uzun menzili unutmayan ortam (%55 yakın, %25 orta, %20 uzak)
    env_ft = build_vec_env(
        n_envs=n_envs,
        base_seed=12000,
        expert_weight=0.0,
        pwpf_prob=1.00,
        cbf_prob=1.00,
        dr_prob=0.10,
        curriculum_probs=(0.45, 0.30, 0.25),
    )

    model = PPO.load(CHAMPION_MODEL_PATH, env=env_ft, device="cpu")
    model.ent_coef = 0.0005
    model.lr_schedule = linear_schedule(3e-5, 5e-6)

    print(f"\n🔥 İnce Ayar Başlatılıyor: {finetune_steps:,} adım (n_envs={n_envs})...")
    t0 = time.time()
    model.learn(total_timesteps=finetune_steps, reset_num_timesteps=False)
    elapsed = time.time() - t0
    print(f"   -> İnce Ayar Tamamlandı ({elapsed:.1f}s)")
    env_ft.close()

    model.save(CANDIDATE_MODEL_PATH)
    print(f"💾 Aday Model Kaydedildi: {CANDIDATE_MODEL_PATH}")

    print("\n" + "=" * 80)
    print("🎯 50 UÇUŞLUK SAF MUJOCO DOĞRULAMA TESTİ")
    print("=" * 80)

    res_raw = evaluate_pure_mujoco(model, mode="pure_raw", num_eval_episodes=50)
    res_pwpf = evaluate_pure_mujoco(model, mode="pure_pwpf", num_eval_episodes=50)
    res_cbf_pwpf = evaluate_pure_mujoco(model, mode="pure_cbf_pwpf", num_eval_episodes=50)

    new_success = float(res_cbf_pwpf["success_rate"])
    if new_success >= prev_success:
        shutil.copy2(CANDIDATE_MODEL_PATH, CHAMPION_MODEL_PATH)
        print(f"\n🏆 YENİ ŞAMPİYON! (%{prev_success:.1f} -> %{new_success:.1f}) -> {CHAMPION_MODEL_PATH} güncellendi.")

        summary_stats = {
            "model_name": "Champion-PPO v4 (Terminal PWPF+CBF Fine-Tuned + Anti-Hover)",
            "total_timesteps": 700000 + finetune_steps,
            "evaluation_episodes": 50,
            "success_rate": res_cbf_pwpf["success_rate"],
            "crash_rate": res_cbf_pwpf["crash_rate"],
            "hover_rate": res_cbf_pwpf["hover_rate"],
            "success_count": res_cbf_pwpf["success_count"],
            "crash_count": res_cbf_pwpf["crash_count"],
            "hover_count": res_cbf_pwpf["hover_count"],
            "mean_final_distance": res_cbf_pwpf["mean_final_distance"],
            "std_final_distance": res_cbf_pwpf["std_final_distance"],
            "mean_final_speed": res_cbf_pwpf["mean_final_speed"],
            "std_final_speed": res_cbf_pwpf["std_final_speed"],
            "mean_delta_v": res_cbf_pwpf["mean_delta_v"],
            "std_delta_v": res_cbf_pwpf["std_delta_v"],
            "target_met": bool(res_cbf_pwpf["success_rate"] >= 70.0),
            "pure_modes_summary": {
                "pure_raw": {k: v for k, v in res_raw.items() if k != "detailed_results"},
                "pure_pwpf": {k: v for k, v in res_pwpf.items() if k != "detailed_results"},
                "pure_cbf_pwpf": {k: v for k, v in res_cbf_pwpf.items() if k != "detailed_results"},
            },
            "detailed_results": res_cbf_pwpf["detailed_results"],
        }
        with open(STATS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"📁 İstatistikler güncellendi: {STATS_OUTPUT_PATH}")
    else:
        print(f"\n⚠️ Aday model (%{new_success:.1f}) mevcut baseline'ı (%{prev_success:.1f}) geçemedi. Champion korunuyor.")


if __name__ == "__main__":
    run_terminal_finetune()
