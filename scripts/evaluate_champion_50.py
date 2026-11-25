"""
Orbital-RL: Şampiyon PPO 50 Uçuşluk Saf MuJoCo Monte Carlo Doğrulama Scripti
----------------------------------------------------------------------------
Analitik GNC devralması OLMADAN, eğitilmiş 700K Champion-PPO modelini
3 saf MuJoCo modunda (pure_raw, pure_pwpf, pure_cbf_pwpf) 50 uçuşta test eder.
"""

import os
import sys
import json
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mujoco_champion_ppo import evaluate_pure_mujoco

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion.zip")
STATS_PATH = os.path.join(DIR_PATH, "mujoco_champion_benchmark_stats.json")


def evaluate_champion():
    print("=" * 80)
    print("ORBITAL-RL // 50-EPISODE PURE MUJOCO MONTE CARLO BENCHMARK (NO SERVO TAKEOVER)")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 80)

    model = PPO.load(MODEL_PATH, device="cpu")

    res_raw = evaluate_pure_mujoco(model, mode="pure_raw", num_eval_episodes=50)
    res_pwpf = evaluate_pure_mujoco(model, mode="pure_pwpf", num_eval_episodes=50)
    res_cbf_pwpf = evaluate_pure_mujoco(model, mode="pure_cbf_pwpf", num_eval_episodes=50)

    summary_stats = {
        "model_name": "Champion-PPO v3 (700K Pure MuJoCo + PWPF/CBF-Aware + X-Z Centered)",
        "total_timesteps": 700000,
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

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nSaved verified stats -> {STATS_PATH}")
    return summary_stats


if __name__ == "__main__":
    evaluate_champion()
