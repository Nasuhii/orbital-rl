"""
Orbital-RL: MuJoCo 3D Simülasyonu ve Telemetri Doğrulayıcısı
-----------------------------------------------------------
Bu script, Google DeepMind MuJoCo fizik motoru ve Clohessy-Wiltshire (CW)
yörüngesel kuvvetleriyle çalışan OrbitalMujocoEnv ortamını test eder,
Asymptotic vs Naive kontrolcülerini yarıştırır ve yüksek çözünürlüklü
telemetri grafiği üretir.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbital_mujoco_env import OrbitalMujocoEnv


def run_mujoco_simulation():
    print("\n" + "=" * 68)
    print("🛰️  ORBITAL-RL: MUJOCO & GYMNASIUM 3D SİMÜLASYONU BAŞLATILIYOR")
    print("=" * 68)
    print("Fizik Motoru : Google DeepMind MuJoCo v3.17 + CW Orbital Kuvvetler")
    print("Ortam API    : Gymnasium v1.0+")
    print("Entegratör   : MuJoCo RK4 (Runge-Kutta 4th Order)")
    print("Adım Süresi  : frame_skip=5 (dt = 0.5 saniye / adım)")
    print("=" * 68 + "\n")

    # 1. Asymptotic Shaped Ortamı
    env_shaped = OrbitalMujocoEnv(use_asymptotic_reward=True, max_steps=220)
    # 2. Naive Doğrusal Ortamı
    env_naive = OrbitalMujocoEnv(use_asymptotic_reward=False, max_steps=220)

    # İki ortamı aynı başlangıç noktasına al
    obs_s, info_s = env_shaped.reset(seed=101)
    obs_n, info_n = env_naive.reset(seed=101)

    env_naive.data.qpos[:] = env_shaped.data.qpos[:].copy()
    env_naive.data.qvel[:] = env_shaped.data.qvel[:].copy()

    traj_s = []
    traj_n = []
    thrust_s = []
    thrust_n = []

    print("▶️  1. Asymptotic Shaped Kontrolcüsü MuJoCo üzerinde simüle ediliyor...")
    done = False
    truncated = False
    while not (done or truncated):
        rel_pos = obs_s[:3]
        vel = obs_s[3:6]
        dist = info_s['distance']
        speed = info_s['speed']

        target_speed = 0.045 + 0.45 * np.tanh(dist / 8.0)
        desired_vel = -rel_pos / (dist + 1e-5) * target_speed
        vel_err = desired_vel - vel

        coriolis_x = -2.0 * env_shaped.n * vel[1] * env_shaped.m
        coriolis_y =  2.0 * env_shaped.n * vel[0] * env_shaped.m
        p_gain = 90.0 if dist > 4.0 else 130.0
        force = (vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0])
        action = np.clip(force / env_shaped.max_thrust, -1.0, 1.0)

        thrust_s.append(action * env_shaped.max_thrust)
        obs_s, reward, done, truncated, info_s = env_shaped.step(action)
        traj_s.append(np.concatenate([obs_s[:3].copy(), obs_s[3:6].copy()]))

    status_s = "🟢 DOCKED (Başarılı)" if info_s['docked_success'] else ("💥 CRASH" if info_s['crashed'] else "⏸️ HOVER")
    print(f"   • Shaped Son Mesafe : {info_s['distance']:.3f} m | Son Hız: {info_s['speed']:.3f} m/s | Durum: {status_s}")

    print("▶️  2. Naive Doğrusal Kontrolcüsü MuJoCo üzerinde simüle ediliyor...")
    done = False
    truncated = False
    while not (done or truncated):
        rel_pos = obs_n[:3]
        vel = obs_n[3:6]
        dist = info_n['distance']
        speed = info_n['speed']

        if dist > 3.0:
            target_vel = -rel_pos / (dist + 1e-5) * 0.26
            action = np.clip((target_vel - vel) * 5.0, -1.0, 1.0)
        else:
            action = np.clip(-vel * 9.0, -1.0, 1.0)

        thrust_n.append(action * env_naive.max_thrust)
        obs_n, reward, done, truncated, info_n = env_naive.step(action)
        traj_n.append(np.concatenate([obs_n[:3].copy(), obs_n[3:6].copy()]))

    status_n = "🟢 DOCKED (Başarılı)" if info_n['docked_success'] else ("💥 CRASH" if info_n['crashed'] else "⏸️ HOVER")
    print(f"   • Naive Son Mesafe  : {info_n['distance']:.3f} m | Son Hız: {info_n['speed']:.3f} m/s | Durum: {status_n}")

    # ==========================================
    # GRAFİK ÇİZİMİ (MuJoCo Telemetri Paneli)
    # ==========================================
    ts = np.array(traj_s)
    tn = np.array(traj_n)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#090d16")

    # 1. Panel: 3D Yörünge
    ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#0d1424")
    ax1.scatter([0], [0], [0], color="#f59e0b", s=200, marker="*", label="İstasyon Limanı (0,0,0)")
    ax1.plot(ts[:, 0], ts[:, 1], ts[:, 2], color="#10b981", linewidth=2.5, label="MuJoCo: Asymptotic Shaped")
    ax1.plot(tn[:, 0], tn[:, 1], tn[:, 2], color="#ef4444", linewidth=2.0, linestyle="--", label="MuJoCo: Naive Doğrusal")
    ax1.set_title("A. MuJoCo 3D Yörüngesel Kenetlenme Rotası", color="#38bdf8", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Radyal X (m)", color="#94a3b8")
    ax1.set_ylabel("Along-Track Y (m)", color="#94a3b8")
    ax1.set_zlabel("Cross-Track Z (m)", color="#94a3b8")
    ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.3)

    # 2. Panel: Mesafe vs Zaman
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#0d1424")
    dist_s = np.linalg.norm(ts[:, :3], axis=1)
    dist_n = np.linalg.norm(tn[:, :3], axis=1)
    ax2.plot(dist_s, color="#10b981", linewidth=2.2, label=f"Shaped (Son: {dist_s[-1]:.2f}m)")
    ax2.plot(dist_n, color="#ef4444", linewidth=1.8, linestyle="--", label=f"Naive (Son: {dist_n[-1]:.2f}m)")
    ax2.axhline(0.25, color="#38bdf8", linestyle=":", label="Kenetlenme Eşiği (0.25m)")
    ax2.set_title("B. MuJoCo İstasyona Mesafe Profili", color="#38bdf8", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Adım (Step = 0.5s)", color="#94a3b8")
    ax2.set_ylabel("Mesafe (m)", color="#94a3b8")
    ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.3)

    # 3. Panel: Bağıl Hız vs Güvenlik Sınırı
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#0d1424")
    speed_s = np.linalg.norm(ts[:, 3:6], axis=1)
    speed_n = np.linalg.norm(tn[:, 3:6], axis=1)
    ax3.plot(speed_s, color="#10b981", linewidth=2.0, label="Shaped Hızı")
    ax3.plot(speed_n, color="#ef4444", linewidth=1.8, linestyle="--", label="Naive Hızı")
    ax3.axhline(0.06, color="#f43f5e", linestyle=":", label="Kaza Hız Eşiği (0.06 m/s)")
    ax3.set_title("C. MuJoCo Bağıl Hız ve Temas Güvenliği", color="#38bdf8", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Adım (Step = 0.5s)", color="#94a3b8")
    ax3.set_ylabel("Hız (m/s)", color="#94a3b8")
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.3)

    # 4. Panel: MuJoCo RCS İtici Kuvvetleri (Shaped Ajan)
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#0d1424")
    fs = np.array(thrust_s)
    ax4.plot(fs[:, 0], color="#ef4444", linewidth=1.3, alpha=0.8, label="Fx (Radyal / Coriolis)")
    ax4.plot(fs[:, 1], color="#10b981", linewidth=1.5, alpha=0.9, label="Fy (Along-Track Fren)")
    ax4.plot(fs[:, 2], color="#38bdf8", linewidth=1.3, alpha=0.8, label="Fz (Cross-Track Düzlem)")
    ax4.axhline(0, color="#64748b", linestyle=":", alpha=0.5)
    ax4.set_title("D. MuJoCo 3 Eksen İtici Aktüatör Kuvvetleri (N)", color="#38bdf8", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Adım (Step = 0.5s)", color="#94a3b8")
    ax4.set_ylabel("İtki (Newton)", color="#94a3b8")
    ax4.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", fontsize=8)
    ax4.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_docking_telemetry.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n[MUJOCO GRAFİK KAYDEDİLDİ] -> {out_path}\n")


if __name__ == "__main__":
    run_mujoco_simulation()
