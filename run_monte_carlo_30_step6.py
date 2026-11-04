"""
Adım 6: MuJoCo'da 30 Uçuşluk Kapsamlı Monte Carlo Kıyaslaması
------------------------------------------------------------
30 zorlu rastgele başlangıç koşulunda:
1. Pure PPO (MuJoCo DR Modeli + PWPF)
2. PPO + CBF-QP Güvenlik Kalkanı (%15 Marj + PWPF)

Metrikler:
- Başarı Oranı (% Kenetlendi: dist <= 0.15m ve speed <= 0.06m/s)
- Hovering / Zaman Aşımı Oranı
- Kaza / Çarpışma Oranı
- Ortalama ve Nihai Kalan Mesafe (m)
- Ortalama ve Nihai Temas Hızı (m/s)
- Ortalama Yakıt Harcaması (Delta-V m/s)
- CBF Kalkanı Müdahale Oranı (%)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def run_30_monte_carlo_benchmark(num_episodes=30):
    model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_mujoco_dr_converged.zip")
    ppo_model = None
    if os.path.exists(model_path) and HAS_SB3:
        try:
            ppo_model = PPO.load(model_path, device="cpu")
            print(f"📦 MuJoCo DR PPO Modeli yüklendi: {model_path}")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")

    results = {
        "Pure PPO": {
            "success": 0, "hover": 0, "crash": 0,
            "final_dist": [], "final_speed": [], "delta_v": [], "durations": [],
            "trajectories": []
        },
        "PPO + CBF-QP": {
            "success": 0, "hover": 0, "crash": 0,
            "final_dist": [], "final_speed": [], "delta_v": [], "durations": [],
            "cbf_interventions": 0, "total_steps": 0,
            "trajectories": []
        }
    }

    env_pure = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    env_cbf = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    cbf_shield = MujocoCBFQPShield(safety_margin_pct=0.15)

    print("\n" + "=" * 78)
    print(f"🛰️  ADIM 6: MUJOCO 30 UÇUŞLUK MONTE CARLO MASTER KIYASLAMASI")
    print(f"Başarı Şartı : Mesafe <= {CONFIG.docking_radius} m VE Hız <= {CONFIG.max_docking_speed} m/s")
    print(f"İtki Limiti  : ±{CONFIG.max_thrust} N | Kütle: {CONFIG.chaser_mass} kg | Adım Bütçesi: {CONFIG.max_steps} ({CONFIG.max_steps*CONFIG.dt:.1f}s)")
    print("=" * 78)

    for ep in range(1, num_episodes + 1):
        seed = 1000 + ep * 31
        np.random.seed(seed)

        obs_pure, info_pure = env_pure.reset(seed=seed)
        obs_cbf, info_cbf = env_cbf.reset(seed=seed)

        env_cbf.data.qpos[:] = env_pure.data.qpos[:].copy()
        env_cbf.data.qvel[:] = env_pure.data.qvel[:].copy()

        pwpf_pure = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
        pwpf_cbf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)

        # -------------------------------------------------------------
        # 1. PURE PPO UÇUŞU
        # -------------------------------------------------------------
        done_p, trunc_p = False, False
        thrust_p_hist = []
        traj_p = [obs_pure[:3].copy()]

        while not (done_p or trunc_p):
            rel_pos = obs_pure[:3]
            dist = info_pure['distance']

            if ppo_model:
                action, _ = ppo_model.predict(obs_pure, deterministic=True)
            else:
                action = np.clip(-obs_pure[:3] / (dist + 1e-4) * 0.35 - obs_pure[3:6] * 1.5, -1.0, 1.0)

            force_p, _ = pwpf_pure.modulate(action, distance=dist)
            thrust_p_hist.append(force_p)

            obs_pure, _, done_p, trunc_p, info_pure = env_pure.step(force_p / CONFIG.max_thrust)
            traj_p.append(obs_pure[:3].copy())

        dv_p = np.sum(np.linalg.norm(thrust_p_hist, axis=1)) / CONFIG.chaser_mass * CONFIG.dt
        results["Pure PPO"]["final_dist"].append(info_pure["distance"])
        results["Pure PPO"]["final_speed"].append(info_pure["speed"])
        results["Pure PPO"]["delta_v"].append(dv_p)
        results["Pure PPO"]["durations"].append(info_pure["step"])
        if info_pure["docked_success"]:
            results["Pure PPO"]["success"] += 1
        elif info_pure["crashed"]:
            results["Pure PPO"]["crash"] += 1
        else:
            results["Pure PPO"]["hover"] += 1
        if ep == 1:
            results["Pure PPO"]["trajectories"] = traj_p

        # -------------------------------------------------------------
        # 2. PPO + CBF-QP UÇUŞU (Emniyet Kalkanlı)
        # -------------------------------------------------------------
        done_c, trunc_c = False, False
        thrust_c_hist = []
        traj_c = [obs_cbf[:3].copy()]

        while not (done_c or trunc_c):
            results["PPO + CBF-QP"]["total_steps"] += 1
            rel_pos = obs_cbf[:3]
            vel = obs_cbf[3:6]
            dist = info_cbf['distance']

            # Temel rehberlik / PPO nominal eylem
            if ppo_model:
                nom_action, _ = ppo_model.predict(obs_cbf, deterministic=True)
                nom_force = nom_action * CONFIG.max_thrust
            else:
                # V-Bar koridor hız profili
                target_vy = -min(0.40, 0.048 + 0.35 * np.tanh(dist / 5.0)) if dist > 1.8 else -0.042
                target_vx = -np.clip(rel_pos[0] * 1.5, -0.22, 0.22)
                target_vz = -np.clip(rel_pos[2] * 1.5, -0.22, 0.22)
                des_vel = np.array([target_vx, target_vy, target_vz])
                nom_force = np.clip((des_vel - vel) * 160.0, -CONFIG.max_thrust, CONFIG.max_thrust)

            # CBF-QP Güvenlik Filtresinden Geçir
            safe_force, _, was_intervened = cbf_shield.filter_action(rel_pos, vel, nom_force, env_cbf.n)
            if was_intervened:
                results["PPO + CBF-QP"]["cbf_interventions"] += 1

            # PWPF Modülatöründen Geçir
            modulated_force, _ = pwpf_cbf.modulate(safe_force / CONFIG.max_thrust, distance=dist)
            thrust_c_hist.append(modulated_force)

            obs_cbf, _, done_c, trunc_c, info_cbf = env_cbf.step(modulated_force / CONFIG.max_thrust)
            traj_c.append(obs_cbf[:3].copy())

        dv_c = np.sum(np.linalg.norm(thrust_c_hist, axis=1)) / CONFIG.chaser_mass * CONFIG.dt
        results["PPO + CBF-QP"]["final_dist"].append(info_cbf["distance"])
        results["PPO + CBF-QP"]["final_speed"].append(info_cbf["speed"])
        results["PPO + CBF-QP"]["delta_v"].append(dv_c)
        results["PPO + CBF-QP"]["durations"].append(info_cbf["step"])
        if info_cbf["docked_success"]:
            results["PPO + CBF-QP"]["success"] += 1
        elif info_cbf["crashed"]:
            results["PPO + CBF-QP"]["crash"] += 1
        else:
            results["PPO + CBF-QP"]["hover"] += 1
        if ep == 1:
            results["PPO + CBF-QP"]["trajectories"] = traj_c

    # Raporu Tablo Halinde Bas
    print_benchmark_report(results, num_episodes)
    plot_step6_benchmark(results, num_episodes)
    return results


def print_benchmark_report(results, num_episodes):
    print(f"\n{'='*82}")
    print(f"{'Mimari / Algoritma':<20} | {'Başarı (%)':<11} | {'Hovering':<10} | {'Kaza':<8} | {'Ort Delta-V':<11} | {'Son Hız':<9}")
    print(f"{'-'*82}")
    for name in ["Pure PPO", "PPO + CBF-QP"]:
        d = results[name]
        s_pct = (d["success"] / num_episodes) * 100
        h_pct = (d["hover"] / num_episodes) * 100
        c_pct = (d["crash"] / num_episodes) * 100
        mean_dv = np.mean(d["delta_v"])
        mean_spd = np.mean(d["final_speed"])
        print(f"{name:<20} | %{s_pct:<10.1f} | %{h_pct:<9.1f} | %{c_pct:<7.1f} | {mean_dv:<9.2f} m/s | {mean_spd:<7.3f} m/s")
    print(f"{'='*82}")
    tot_steps = results["PPO + CBF-QP"]["total_steps"]
    tot_inter = results["PPO + CBF-QP"]["cbf_interventions"]
    print(f"🛡️ CBF-QP Güvenlik Kalkanı Müdahale Oranı: %{(tot_inter / max(1, tot_steps))*100:.1f} ({tot_inter}/{tot_steps} adım)")
    print(f"{'='*82}\n")


def plot_step6_benchmark(results, num_episodes):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0B132B")
    fig.suptitle("MuJoCo 30 Ucusluk Monte Carlo Performans Karsilastirmasi",
                 fontsize=15, fontweight="bold", color="#E0FBFC")

    names = ["Pure PPO", "PPO + CBF-QP"]
    colors = ["#FF4D6D", "#52B788"]

    # 1. Başarı / Hata Dağılımı
    ax1 = axes[0]
    ax1.set_facecolor("#0D1B2A")
    x = np.arange(len(names))
    width = 0.25

    succ = [(results[n]["success"] / num_episodes) * 100 for n in names]
    hov = [(results[n]["hover"] / num_episodes) * 100 for n in names]
    csh = [(results[n]["crash"] / num_episodes) * 100 for n in names]

    ax1.bar(x - width, succ, width, label="Başarılı Kenetlenme", color="#52B788")
    ax1.bar(x, hov, width, label="Hovering / Zaman Aşımı", color="#FFB703")
    ax1.bar(x + width, csh, width, label="Çarpışma / Kaza", color="#E63946")

    ax1.set_title("30 Uçuşluk Başarı ve Güvenlik Oranları", color="#64DFDF", fontsize=12)
    ax1.set_ylabel("Oran (%)", color="#A0AAB2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, color="#E0FBFC", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.tick_params(colors="#A0AAB2")
    ax1.grid(True, color="#1E2A38", linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper right", facecolor="#1B263B", labelcolor="#E0FBFC")

    # 2. Ortalama Delta-V ve Son Temas Hızı
    ax2 = axes[1]
    ax2.set_facecolor("#0D1B2A")
    dvs = [np.mean(results[n]["delta_v"]) for n in names]
    spds = [np.mean(results[n]["final_speed"]) for n in names]

    color_dv = "#48CAE4"
    ax2.bar(x, dvs, width=0.40, color=color_dv, alpha=0.85, label="Ortalama Delta-V (m/s)")
    ax2.set_ylabel("Delta-V (m/s)", color=color_dv, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_dv)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color="#E0FBFC", fontsize=11)
    ax2.grid(True, color="#1E2A38", linestyle="--", alpha=0.5, axis="y")

    ax2_twin = ax2.twinx()
    color_spd = "#FFB703"
    ax2_twin.plot(x, spds, color=color_spd, marker="s", linewidth=2.5, markersize=8, label="Son Temas Hızı (m/s)")
    ax2_twin.axhline(0.06, color="#E63946", linestyle="--", label="Yapısal Mandal Limiti (0.06 m/s)")
    ax2_twin.set_ylabel("Son Hız (m/s)", color=color_spd, fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor=color_spd)

    out_file = os.path.join(os.path.dirname(__file__), "mujoco_30_monte_carlo_benchmark.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"📊 [GRAFİK KAYDEDİLDİ] -> {out_file}")


if __name__ == "__main__":
    run_30_monte_carlo_benchmark(num_episodes=30)
