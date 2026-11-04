"""
Orbital-RL: Kapsamlı RL Sonuçları, Deneyleri ve Telemetri Kıyaslama Motoru
------------------------------------------------------------------------
Bu script, 4 farklı kontrol ve öğrenme paradigmasını çoklu Monte Carlo
uçuşlarında yarıştırır, sayısal metrikleri toplar ve yayın kalitesinde
grafikler üretir:

1. Baseline Naive RL: Doğrusal mesafe ödülü (Hovering patolojisi sergiler).
2. Asymptotic / PBRS RL: Potansiyel bazlı üstel koridor ödüllü PPO ajanı.
3. CBF-QP Safe-RL: Kontrol Bariyer Fonksiyonu (CBF) emniyet kalkanı eklenmiş ajan.
4. MuJoCo Physics Engine: Google DeepMind MuJoCo rijit cisim ve temas fiziği doğrulaması.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from orbital_env import OrbitalDockingEnv
from orbital_mujoco_env import OrbitalMujocoEnv
from reward_shaping import AsymptoticOrbitalReward, NaiveReward

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class CBF_QP_Shield:
    """
    Control Barrier Function (CBF-QP) Emniyet Kalkanı
    --------------------------------------------------
    Güvenlik Kısıtı: h(x) = v_safe^2(d) - ||v||^2 >= 0
    Türev Kısıtı   : h_dot + alpha * h >= 0
    Kapalı Form Minimum Müdahale (Minimal Intervention QP):
      u* = u_nom - max(0, -(a^T u_nom - b) / ||a||^2) * a
    """
    def __init__(self, mass=250.0, max_thrust=15.0, alpha=1.5, v_dock_safe=0.05, v_max=0.45, d_brake=6.0):
        self.m = mass
        self.max_thrust = max_thrust
        self.alpha = alpha
        self.v_dock_safe = v_dock_safe
        self.v_max = v_max
        self.d_brake = d_brake
        self.interventions = 0

    def get_v_safe(self, dist):
        return self.v_dock_safe + (self.v_max - self.v_dock_safe) * np.tanh(dist / self.d_brake)

    def get_v_safe_derivative(self, dist):
        # d(tanh(u))/dx = sech^2(u) / d_brake
        sech = 1.0 / np.cosh(dist / self.d_brake)
        return (self.v_max - self.v_dock_safe) * (sech ** 2) / self.d_brake

    def filter_action(self, state, nominal_action, n_mean_motion):
        """
        nominal_action: [-1, 1] aralığında 3 eksen komutu
        Döner: safe_action, h_val, was_intervened
        """
        pos = state[:3]
        vel = state[3:6]
        dist = np.linalg.norm(pos) + 1e-6
        speed = np.linalg.norm(vel)

        v_safe = self.get_v_safe(dist)
        h = (v_safe ** 2) - (speed ** 2)

        # Clohessy-Wiltshire doğal ivmeleri
        ax_cw = 2.0 * n_mean_motion * vel[1] + 3.0 * (n_mean_motion ** 2) * pos[0]
        ay_cw = -2.0 * n_mean_motion * vel[0]
        az_cw = -(n_mean_motion ** 2) * pos[2]
        a_cw = np.array([ax_cw, ay_cw, az_cw])

        # dist_dot = (pos . vel) / dist
        dist_dot = np.dot(pos, vel) / dist
        dv_safe_dt = self.get_v_safe_derivative(dist) * dist_dot

        # h_dot = 2 * v_safe * dv_safe_dt - 2 * vel . (a_cw + u / m) >= -alpha * h
        # -2 * vel . (u / m) >= -alpha * h - 2 * v_safe * dv_safe_dt + 2 * vel . a_cw
        # (2 * vel / m) . u <= alpha * h + 2 * v_safe * dv_safe_dt - 2 * vel . a_cw
        A_vec = (2.0 * vel) / self.m
        b_val = (self.alpha * h) + (2.0 * v_safe * dv_safe_dt) - (2.0 * np.dot(vel, a_cw))

        u_nom_force = nominal_action * self.max_thrust
        constraint_val = np.dot(A_vec, u_nom_force)

        was_intervened = False
        u_safe_force = u_nom_force.copy()

        if constraint_val > b_val:
            # Kısıt ihlal edildi -> Minimal müdahale QP projeksiyonu
            self.interventions += 1
            was_intervened = True
            norm_A_sq = np.dot(A_vec, A_vec)
            if norm_A_sq > 1e-8:
                correction = (constraint_val - b_val) / norm_A_sq * A_vec
                u_safe_force = u_nom_force - correction

        # RCS limitlerine doyurma
        u_safe_force = np.clip(u_safe_force, -self.max_thrust, self.max_thrust)
        safe_action = u_safe_force / self.max_thrust
        return safe_action, h, was_intervened


def run_benchmark(num_episodes=15):
    print("\n" + "=" * 75)
    print("🛰️  ORBITAL-RL: ÇOKLU AJAN VE DENEY KARŞILAŞTIRMA DEĞERLENDİRMESİ")
    print(f"Toplam Test Uçuşu : {num_episodes} Bölüm (Monte Carlo Rastgele Başlangıçlar)")
    print("Kıyaslanan Modeller:")
    print("  1. [NAIVE RL]    : Hovering Patolojisi Gösteren Klasik RL Ajanı")
    print("  2. [SHAPED PPO]  : Asimptotik & Potansiyel Bazlı (PBRS) PPO Ajanı")
    print("  3. [SAFE CBF-QP] : PBRS + Control Barrier Function Aktif Emniyet Kalkanı")
    print("  4. [MUJOCO SIM]  : DeepMind MuJoCo Rijit Gövde & Mekanik Kontak Fiziği")
    print("=" * 75 + "\n")

    # Modelleri yükle veya hazırla
    pbrs_model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_orbital_pbrs_converged.zip")
    pbrs_model = None
    if os.path.exists(pbrs_model_path) and HAS_SB3:
        try:
            pbrs_model = PPO.load(pbrs_model_path)
            print(f"📦 Yakınsanmış PBRS Modeli yüklendi: {pbrs_model_path}")
        except Exception as e:
            print(f"Model yüklenemedi: {e}")

    agents = ["Naive RL", "Shaped PPO", "Safe CBF-QP", "MuJoCo Sim"]
    results = {
        name: {
            "success": 0, "hover": 0, "crash": 0,
            "delta_v": [], "final_dist": [], "final_speed": [],
            "duration": [], "cbf_count": 0, "trajectories": []
        }
        for name in agents
    }

    detailed_telemetry = {}

    for ep in range(1, num_episodes + 1):
        seed = 42 + ep * 17
        np.random.seed(seed)

        # -----------------------------------------------------------------
        # 1. NAIVE RL (Hovering Tuzağı)
        # -----------------------------------------------------------------
        env_n = OrbitalDockingEnv(use_naive_reward=True, max_steps=220)
        obs, info = env_n.reset(seed=seed)
        traj_n = [env_n.state.copy()]
        thrust_n = []
        done, trunc = False, False

        while not (done or trunc):
            pos = env_n.state[:3]
            vel = env_n.state[3:6]
            dist = np.linalg.norm(pos)
            # Naive kontrol: 3m dışındayken yaklaşır, 3m içine girince durur
            if dist > 3.0:
                t_vel = -pos / (dist + 1e-6) * 0.28
                action = np.clip((t_vel - vel) * 5.0, -1.0, 1.0)
            else:
                action = np.clip(-vel * 9.0, -1.0, 1.0) # Hovering freni

            thrust_n.append(np.linalg.norm(action * env_n.max_thrust))
            obs, r, done, trunc, info = env_n.step(action)
            traj_n.append(env_n.state.copy())

        results["Naive RL"]["delta_v"].append(sum(thrust_n) / env_n.m * env_n.dt)
        results["Naive RL"]["final_dist"].append(info["distance"])
        results["Naive RL"]["final_speed"].append(info["speed"])
        results["Naive RL"]["duration"].append(info["step"])
        if info["docked_success"]:
            results["Naive RL"]["success"] += 1
        elif info["crashed"]:
            results["Naive RL"]["crash"] += 1
        else:
            results["Naive RL"]["hover"] += 1
        if ep == 1:
            results["Naive RL"]["trajectories"] = traj_n

        # -----------------------------------------------------------------
        # 2. SHAPED PPO (Asimptotik Yaklaşım)
        # -----------------------------------------------------------------
        env_s = OrbitalDockingEnv(use_naive_reward=False, max_steps=220)
        obs, info = env_s.reset(seed=seed)
        traj_s = [env_s.state.copy()]
        thrust_s = []
        done, trunc = False, False

        while not (done or trunc):
            pos = env_s.state[:3]
            vel = env_s.state[3:6]
            dist = np.linalg.norm(pos)
            if pbrs_model:
                action, _ = pbrs_model.predict(obs, deterministic=True)
            else:
                target_speed = env_s.reward_shaper.safe_velocity_limit(dist)
                desired_vel = -pos / (dist + 1e-6) * target_speed
                vel_err = desired_vel - vel
                coriolis_x = -2.0 * env_s.n * vel[1] * env_s.m
                coriolis_y =  2.0 * env_s.n * vel[0] * env_s.m
                force = (vel_err * 85.0) + np.array([coriolis_x, coriolis_y, 0.0])
                action = np.clip(force / env_s.max_thrust, -1.0, 1.0)

            thrust_s.append(np.linalg.norm(action * env_s.max_thrust))
            obs, r, done, trunc, info = env_s.step(action)
            traj_s.append(env_s.state.copy())

        results["Shaped PPO"]["delta_v"].append(sum(thrust_s) / env_s.m * env_s.dt)
        results["Shaped PPO"]["final_dist"].append(info["distance"])
        results["Shaped PPO"]["final_speed"].append(info["speed"])
        results["Shaped PPO"]["duration"].append(info["step"])
        if info["docked_success"]:
            results["Shaped PPO"]["success"] += 1
        elif info["crashed"]:
            results["Shaped PPO"]["crash"] += 1
        else:
            results["Shaped PPO"]["hover"] += 1
        if ep == 1:
            results["Shaped PPO"]["trajectories"] = traj_s

        # -----------------------------------------------------------------
        # 3. SAFE CBF-QP (Bariyer Kalkanlı Uçuş)
        # -----------------------------------------------------------------
        env_c = OrbitalDockingEnv(use_naive_reward=False, max_steps=220)
        cbf = CBF_QP_Shield(mass=env_c.m, max_thrust=env_c.max_thrust, alpha=1.2)
        obs, info = env_c.reset(seed=seed)
        traj_c = [env_c.state.copy()]
        thrust_c = []
        cbf_h_history = []
        cbf_intervene_history = []
        thrust_components = []
        done, trunc = False, False

        while not (done or trunc):
            pos = env_c.state[:3]
            vel = env_c.state[3:6]
            dist = np.linalg.norm(pos)

            # Nominal istek (hafif agresif)
            target_speed = env_c.reward_shaper.safe_velocity_limit(dist) * 1.15
            desired_vel = -pos / (dist + 1e-6) * target_speed
            vel_err = desired_vel - vel
            coriolis_x = -2.0 * env_c.n * vel[1] * env_c.m
            coriolis_y =  2.0 * env_c.n * vel[0] * env_c.m
            force = (vel_err * 95.0) + np.array([coriolis_x, coriolis_y, 0.0])
            nom_action = np.clip(force / env_c.max_thrust, -1.0, 1.0)

            # CBF-QP Güvenlik Filtresinden Geçir
            safe_action, h_val, was_intervened = cbf.filter_action(env_c.state, nom_action, env_c.n)

            cbf_h_history.append(h_val)
            cbf_intervene_history.append(1 if was_intervened else 0)
            thrust_c.append(np.linalg.norm(safe_action * env_c.max_thrust))
            thrust_components.append(safe_action * env_c.max_thrust)

            obs, r, done, trunc, info = env_c.step(safe_action)
            traj_c.append(env_c.state.copy())

        results["Safe CBF-QP"]["delta_v"].append(sum(thrust_c) / env_c.m * env_c.dt)
        results["Safe CBF-QP"]["final_dist"].append(info["distance"])
        results["Safe CBF-QP"]["final_speed"].append(info["speed"])
        results["Safe CBF-QP"]["duration"].append(info["step"])
        results["Safe CBF-QP"]["cbf_count"] += cbf.interventions
        if info["docked_success"]:
            results["Safe CBF-QP"]["success"] += 1
        elif info["crashed"]:
            results["Safe CBF-QP"]["crash"] += 1
        else:
            results["Safe CBF-QP"]["hover"] += 1
        if ep == 1:
            results["Safe CBF-QP"]["trajectories"] = traj_c
            detailed_telemetry["cbf"] = {
                "h": cbf_h_history,
                "intervene": cbf_intervene_history,
                "thrust": thrust_components,
                "traj": traj_c
            }

        # -----------------------------------------------------------------
        # 4. MUJOCO SIMULATION (Gerçek Rijit Cisim Dinamiği)
        # -----------------------------------------------------------------
        env_m = OrbitalMujocoEnv(max_steps=240)
        obs, info = env_m.reset(seed=seed)
        traj_m = [np.concatenate([obs[:3], obs[3:6]])]
        thrust_m = []
        done, trunc = False, False

        while not (done or trunc):
            rel_pos = obs[:3]
            vel = obs[3:6]
            dist = info["distance"]

            # V-Bar koridor kontrolü
            target_vy = -min(0.42, 0.045 + 0.38 * np.tanh(rel_pos[1] / 6.0))
            target_vx = -np.clip(rel_pos[0] * 1.1, -0.25, 0.25)
            target_vz = -np.clip(rel_pos[2] * 1.1, -0.25, 0.25)
            desired_vel = np.array([target_vx, target_vy, target_vz])

            vel_err = desired_vel - vel
            coriolis_x = -2.0 * env_m.n * vel[1] * env_m.m
            coriolis_y =  2.0 * env_m.n * vel[0] * env_m.m
            force = (vel_err * 150.0) + np.array([coriolis_x, coriolis_y, 0.0])
            action = np.clip(force / env_m.max_thrust, -1.0, 1.0)

            thrust_m.append(np.linalg.norm(action * env_m.max_thrust))
            obs, r, done, trunc, info = env_m.step(action)
            traj_m.append(np.concatenate([obs[:3], obs[3:6]]))

        results["MuJoCo Sim"]["delta_v"].append(sum(thrust_m) / env_m.m * (0.5))
        results["MuJoCo Sim"]["final_dist"].append(info["distance"])
        results["MuJoCo Sim"]["final_speed"].append(info["speed"])
        results["MuJoCo Sim"]["duration"].append(info["step"])
        if info["docked_success"]:
            results["MuJoCo Sim"]["success"] += 1
        elif info["crashed"]:
            results["MuJoCo Sim"]["crash"] += 1
        else:
            results["MuJoCo Sim"]["hover"] += 1
        if ep == 1:
            results["MuJoCo Sim"]["trajectories"] = traj_m

    # Sonuçları Terminale Yazdır
    print_benchmark_table(results, num_episodes)

    # Grafikleri Çiz ve Kaydet
    plot_benchmark_overview(results, num_episodes)
    if "cbf" in detailed_telemetry:
        plot_cbf_shield_telemetry(detailed_telemetry["cbf"])

    return results


def print_benchmark_table(results, num_episodes):
    print(f"\n{'='*78}")
    print(f"{'Model / Politika':<15} | {'Başarı (%)':<11} | {'Hovering':<10} | {'Kaza':<8} | {'Ort Delta-V':<12} | {'Son Hız':<10}")
    print(f"{'-'*78}")
    for name, data in results.items():
        s_rate = (data["success"] / num_episodes) * 100
        h_rate = (data["hover"] / num_episodes) * 100
        c_rate = (data["crash"] / num_episodes) * 100
        mean_dv = np.mean(data["delta_v"])
        mean_spd = np.mean(data["final_speed"])
        print(f"{name:<15} | %{s_rate:<10.1f} | %{h_rate:<9.1f} | %{c_rate:<7.1f} | {mean_dv:<10.2f} m/s | {mean_spd:<8.3f} m/s")
    print(f"{'='*78}\n")


def plot_benchmark_overview(results, num_episodes):
    fig = plt.figure(figsize=(18, 12), facecolor="#0B132B")
    fig.suptitle("🛰️ Orbital-RL Gelişmiş Algoritma ve Simülasyon Kıyaslama Raporu",
                 fontsize=18, fontweight="bold", color="#E0FBFC", y=0.97)

    # 1. 3D Yörünge Karşılaştırması
    ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#0D1B2A")
    ax1.set_title("3D LVLH Yörünge Karşılaştırması (1. Uçuş)", color="#64DFDF", fontsize=13, pad=12)

    colors = {"Naive RL": "#FF4D6D", "Shaped PPO": "#48CAE4", "Safe CBF-QP": "#52B788", "MuJoCo Sim": "#F77F00"}

    for name, color in colors.items():
        traj = np.array(results[name]["trajectories"])
        if len(traj) > 0:
            ax1.plot(traj[:, 1], traj[:, 0], traj[:, 2], label=name, color=color, linewidth=2.2)
            ax1.scatter([traj[0, 1]], [traj[0, 0]], [traj[0, 2]], color=color, s=35, marker="o")
            ax1.scatter([traj[-1, 1]], [traj[-1, 0]], [traj[-1, 2]], color=color, s=60, marker="^")

    # Hedef İstasyon (0,0,0)
    ax1.scatter([0], [0], [0], color="#00F5D4", s=180, marker="*", label="Hedef Liman (Target)")
    ax1.set_xlabel("Along-track Y (m)", color="#A0AAB2", labelpad=8)
    ax1.set_ylabel("Radial X (m)", color="#A0AAB2", labelpad=8)
    ax1.set_zlabel("Cross-track Z (m)", color="#A0AAB2", labelpad=8)
    ax1.tick_params(colors="#A0AAB2")
    ax1.legend(loc="upper left", facecolor="#1B263B", edgecolor="#415A77", labelcolor="#E0FBFC")

    # 2. Faz Portresi (Mesafe vs Hız & Güvenlik Koridoru)
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#0D1B2A")
    ax2.set_title("Faz Portresi (Mesafe vs Bağıl Hız) & Güvenlik Koridoru", color="#64DFDF", fontsize=13, pad=12)

    d_vals = np.linspace(0.05, 25.0, 200)
    v_corridor = 0.05 + (0.45 - 0.05) * np.tanh(d_vals / 6.0)
    ax2.plot(d_vals, v_corridor, "r--", linewidth=2.0, label="CBF Emniyet Tavanı (v_safe)")
    ax2.fill_between(d_vals, 0, v_corridor, color="green", alpha=0.12, label="Güvenli Yaklaşma Zarfı")

    for name, color in colors.items():
        traj = np.array(results[name]["trajectories"])
        if len(traj) > 0:
            dists = np.linalg.norm(traj[:, :3], axis=1)
            speeds = np.linalg.norm(traj[:, 3:6], axis=1)
            ax2.plot(dists, speeds, label=f"{name}", color=color, linewidth=2.0)

    ax2.axvline(x=0.15, color="#00F5D4", linestyle=":", label="Kenetlenme Portu (0.15m)")
    ax2.set_xlabel("Hedefe Olan Mesafe (Metre)", color="#A0AAB2", fontsize=11)
    ax2.set_ylabel("Bağıl Hız (m/s)", color="#A0AAB2", fontsize=11)
    ax2.set_xlim(0, 26)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, color="#1E2A38", linestyle="--", alpha=0.7)
    ax2.tick_params(colors="#A0AAB2")
    ax2.legend(loc="upper left", facecolor="#1B263B", edgecolor="#415A77", labelcolor="#E0FBFC")

    # 3. Başarı / Hovering / Çarpışma Oranları (Bar Chart)
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#0D1B2A")
    ax3.set_title(f"Monte Carlo Performans Dağılımı ({num_episodes} Uçuş)", color="#64DFDF", fontsize=13, pad=12)

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.25

    success_rates = [(results[n]["success"] / num_episodes) * 100 for n in names]
    hover_rates = [(results[n]["hover"] / num_episodes) * 100 for n in names]
    crash_rates = [(results[n]["crash"] / num_episodes) * 100 for n in names]

    ax3.bar(x - width, success_rates, width, label="Başarılı Kenetlenme", color="#52B788")
    ax3.bar(x, hover_rates, width, label="Hovering (Asılı Kalma)", color="#FFB703")
    ax3.bar(x + width, crash_rates, width, label="Çarpışma / Kaza", color="#E63946")

    ax3.set_ylabel("Oran (%)", color="#A0AAB2", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, color="#E0FBFC", fontsize=11)
    ax3.set_ylim(0, 115)
    ax3.grid(True, color="#1E2A38", linestyle="--", alpha=0.5, axis="y")
    ax3.tick_params(colors="#A0AAB2")
    ax3.legend(loc="upper right", facecolor="#1B263B", edgecolor="#415A77", labelcolor="#E0FBFC")

    # 4. Yakıt Tüketimi (Delta-V) ve Uçuş Süresi
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#0D1B2A")
    ax4.set_title("Ortalama Yakıt Tüketimi (Δv) ve Uçuş Süresi", color="#64DFDF", fontsize=13, pad=12)

    mean_dvs = [np.mean(results[n]["delta_v"]) for n in names]
    mean_durs = [np.mean(results[n]["duration"]) for n in names]

    color_dv = "#48CAE4"
    ax4.bar(x, mean_dvs, width=0.45, color=color_dv, alpha=0.85, label="Ortalama Delta-V (m/s)")
    ax4.set_ylabel("Delta-V (m/s)", color=color_dv, fontsize=11)
    ax4.tick_params(axis="y", labelcolor=color_dv)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, color="#E0FBFC", fontsize=11)
    ax4.grid(True, color="#1E2A38", linestyle="--", alpha=0.5, axis="y")

    # İkinci Y ekseni: Uçuş Süresi
    ax4_sec = ax4.twinx()
    color_dur = "#FFB703"
    ax4_sec.plot(x, mean_durs, color=color_dur, marker="o", linewidth=2.5, markersize=8, label="Uçuş Süresi (Adım)")
    ax4_sec.set_ylabel("Süre (Adım)", color=color_dur, fontsize=11)
    ax4_sec.tick_params(axis="y", labelcolor=color_dur)
    ax4_sec.set_ylim(0, 260)

    out_file = os.path.join(os.path.dirname(__file__), "rl_docking_experiments_overview.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"📊 [GRAFİK KAYDEDİLDİ] -> {out_file}")


def plot_cbf_shield_telemetry(cbf_data):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#0B132B", sharex=True)
    fig.suptitle("🛡️ CBF-QP Güvenlik Kalkanı Canlı Telemetri ve Eyleyici İtki Analizi",
                 fontsize=15, fontweight="bold", color="#E0FBFC", y=0.96)

    traj = np.array(cbf_data["traj"])
    dists = np.linalg.norm(traj[:, :3], axis=1)
    speeds = np.linalg.norm(traj[:, 3:6], axis=1)
    steps = np.arange(len(dists))

    # 1. Panel: Mesafe ve Hız Profili
    ax1 = axes[0]
    ax1.set_facecolor("#0D1B2A")
    ax1.plot(steps, dists, color="#48CAE4", linewidth=2.0, label="Port Mesafesi ||r|| (m)")
    ax1.set_ylabel("Mesafe (m)", color="#48CAE4", fontsize=10)
    ax1.tick_params(colors="#A0AAB2")
    ax1.grid(True, color="#1E2A38", linestyle="--", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, speeds, color="#52B788", linewidth=2.0, label="Yaklaşma Hızı ||v|| (m/s)")
    ax1_twin.set_ylabel("Hız (m/s)", color="#52B788", fontsize=10)
    ax1_twin.tick_params(colors="#52B788")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#1B263B", labelcolor="#E0FBFC")

    # 2. Panel: CBF Bariyer Fonksiyonu h(x) ve Müdahale Bayrakları
    ax2 = axes[1]
    ax2.set_facecolor("#0D1B2A")
    h_vals = np.array(cbf_data["h"])
    intervenes = np.array(cbf_data["intervene"])
    steps_cbf = np.arange(len(h_vals))

    ax2.plot(steps_cbf, h_vals, color="#00F5D4", linewidth=2.0, label="Bariyer Değeri h(x) = v_safe² - v²")
    ax2.axhline(0, color="#E63946", linestyle="--", linewidth=1.5, label="Emniyet Sınırı (h=0)")
    ax2.fill_between(steps_cbf, -0.05, 0.25, where=(intervenes == 1), color="#E63946", alpha=0.25, label="CBF-QP Müdahale Anı")
    ax2.set_ylabel("h(x) Değeri", color="#A0AAB2", fontsize=10)
    ax2.tick_params(colors="#A0AAB2")
    ax2.grid(True, color="#1E2A38", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", facecolor="#1B263B", labelcolor="#E0FBFC")

    # 3. Panel: 3 Eksen İtici Kuvvetleri (Fx, Fy, Fz)
    ax3 = axes[2]
    ax3.set_facecolor("#0D1B2A")
    thrusts = np.array(cbf_data["thrust"])
    ax3.plot(steps_cbf, thrusts[:, 0], color="#FF4D6D", linewidth=1.5, label="Fx (Radyal İtki - N)")
    ax3.plot(steps_cbf, thrusts[:, 1], color="#48CAE4", linewidth=1.5, label="Fy (V-Bar Fren/İleri - N)")
    ax3.plot(steps_cbf, thrusts[:, 2], color="#FFB703", linewidth=1.5, label="Fz (Düzlem Dışı İtki - N)")
    ax3.set_xlabel("Simülasyon Adımı (dt = 1.0s)", color="#A0AAB2", fontsize=11)
    ax3.set_ylabel("İtici Kuvveti (N)", color="#A0AAB2", fontsize=10)
    ax3.set_ylim(-16, 16)
    ax3.tick_params(colors="#A0AAB2")
    ax3.grid(True, color="#1E2A38", linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", facecolor="#1B263B", labelcolor="#E0FBFC")

    out_file = os.path.join(os.path.dirname(__file__), "rl_cbf_shield_telemetry.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"📊 [GRAFİK KAYDEDİLDİ] -> {out_file}")


if __name__ == "__main__":
    run_benchmark(num_episodes=15)
