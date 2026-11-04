"""
Orbital-RL: 3D Çok Açılı Detaylı Yörünge ve Vektör Analiz Görselleştiricisi
-------------------------------------------------------------------------
Uzay aracı kenetlenme dinamiğini 6 farklı derin mühendislik perspektifinden
görselleştirir:
  1. 3D İzometrik Yaklaşma Rotası (Coriolis & V-Bar Koridoru ile)
  2. X-Y Düzlemi (Radyal - Along Track / Yerçekimi Gradyanı Etkisi)
  3. Z-Y Düzlemi (Cross-Track Düzlem Dışı Salınım ve Frenleme)
  4. Faz Portresi (Mesafe vs Hız - Asymptotic Tanjant Güvenlik Koridoru)
  5. 3 Eksenli RCS İtki Dağılımı (Fx, Fy, Fz İtici Kuvvetleri)
  6. Asimptotik Ödül ve Çekim Potansiyeli Eğrisi
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon, Circle

from orbital_env import OrbitalDockingEnv
from reward_shaping import AsymptoticOrbitalReward, NaiveReward


def generate_deep_dive_plots():
    print("\n" + "=" * 65)
    print("🛰️  ORBITAL-RL: 6 PANELLİ 3D DERİNLEMESİNE GÖRSEL ANALİZ ÜRETİLİYOR")
    print("=" * 65)

    # 1. Her iki ajan için simülasyonu çalıştır
    env_shaped = OrbitalDockingEnv(use_naive_reward=False, max_steps=220)
    env_naive = OrbitalDockingEnv(use_naive_reward=True, max_steps=220)

    # Aynı başlangıç durumu
    obs_s, _ = env_shaped.reset(seed=42)
    obs_n, _ = env_naive.reset(seed=42)
    env_naive.state = env_shaped.state.copy()
    env_naive.prev_state = env_shaped.state.copy()

    traj_shaped = [env_shaped.state.copy()]
    traj_naive = [env_naive.state.copy()]
    thrust_shaped = []
    thrust_naive = []
    rewards_shaped = []
    rewards_naive = []

    # Asymptotic simülasyonu
    done = False
    trunc = False
    while not (done or trunc):
        pos = env_shaped.state[:3]
        vel = env_shaped.state[3:6]
        dist = np.linalg.norm(pos)
        target_speed = env_shaped.reward_shaper.safe_velocity_limit(dist)
        desired_vel = -pos / (dist + 1e-6) * target_speed
        vel_error = desired_vel - vel
        coriolis_x = -2.0 * env_shaped.n * vel[1] * env_shaped.m
        coriolis_y = 2.0 * env_shaped.n * vel[0] * env_shaped.m
        force = (vel_error * 75.0) + np.array([coriolis_x, coriolis_y, 0.0])
        action = np.clip(force / env_shaped.max_thrust, -1.0, 1.0)
        thrust_shaped.append(action * env_shaped.max_thrust)

        obs, r, done, trunc, info = env_shaped.step(action)
        traj_shaped.append(env_shaped.state.copy())
        rewards_shaped.append(r)

    # Naive simülasyonu
    done = False
    trunc = False
    while not (done or trunc):
        pos = env_naive.state[:3]
        vel = env_naive.state[3:6]
        dist = np.linalg.norm(pos)
        if dist > 3.0:
            target_vel = -pos / (dist + 1e-6) * 0.25
            action = np.clip((target_vel - vel) * 5.0, -1.0, 1.0)
        else:
            action = np.clip(-vel * 8.0, -1.0, 1.0)
        thrust_naive.append(action * env_naive.max_thrust)

        obs, r, done, trunc, info = env_naive.step(action)
        traj_naive.append(env_naive.state.copy())
        rewards_naive.append(r)

    ts = np.array(traj_shaped)
    tn = np.array(traj_naive)
    fs = np.array(thrust_shaped)
    fn = np.array(thrust_naive)

    # ==========================================
    # GRAFİK ÇİZİMİ (6 PANEL - Ultra High Res)
    # ==========================================
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 14), facecolor='#090d16')
    plt.subplots_adjust(left=0.05, right=0.96, top=0.93, bottom=0.06, wspace=0.25, hspace=0.30)

    fig.suptitle("🛰️ ORBITAL-RL: 3D YÖRÜNGESEL KENETLENME DERİNLEMESİNE DİNAMİK VE VEKTÖR ANALİZİ",
                 fontsize=18, fontweight='bold', color='#38bdf8', y=0.98)

    # ----------------------------------------------------
    # PANEL 1: 3D İzometrik Yörünge ve Kenetlenme Konisi
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1, projection='3d', facecolor='#0d1424')
    ax1.scatter([0], [0], [0], color='#f59e0b', s=250, marker='*', label='İstasyon Limanı (0,0,0)', edgecolors='white', zorder=10)

    # 3D Kenetlenme Konisi (LOS Docking Cone)
    cone_len = 22.0
    cone_angle = np.radians(25)
    r_cone = cone_len * np.tan(cone_angle)
    y_grid = np.linspace(0, cone_len, 15)
    theta_grid = np.linspace(0, 2*np.pi, 20)
    Y_c, T_c = np.meshgrid(y_grid, theta_grid)
    R_c = Y_c * np.tan(cone_angle)
    X_c = R_c * np.cos(T_c)
    Z_c = R_c * np.sin(T_c)
    ax1.plot_wireframe(X_c, Y_c, Z_c, color='#38bdf8', alpha=0.15, linewidth=0.6)

    # Yörüngeler
    ax1.plot(ts[:, 0], ts[:, 1], ts[:, 2], color='#10b981', linewidth=2.8, label='Shaped (Kenetlendi)')
    ax1.plot(tn[:, 0], tn[:, 1], tn[:, 2], color='#ef4444', linewidth=2.0, linestyle='--', label='Naive (Hovering)')

    # Vektör Okları: Belirli aralıklarla hız vektörlerini çiz
    for idx in range(0, len(ts)-5, 25):
        ax1.quiver(ts[idx, 0], ts[idx, 1], ts[idx, 2],
                   ts[idx, 3]*10, ts[idx, 4]*10, ts[idx, 5]*10,
                   color='#6ee7b7', length=1.2, arrow_length_ratio=0.3, alpha=0.8)

    ax1.set_title("1. 3D İzometrik Yaklaşma & Vektör Akışı", color='#38bdf8', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("X: Radyal (m)", color='#94a3b8')
    ax1.set_ylabel("Y: Along-Track (m)", color='#94a3b8')
    ax1.set_zlabel("Z: Cross-Track (m)", color='#94a3b8')
    ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.25)

    # ----------------------------------------------------
    # PANEL 2: X-Y Düzlemi (Radyal - Along Track / Coriolis)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2, facecolor='#0d1424')
    ax2.plot(ts[:, 1], ts[:, 0], color='#10b981', linewidth=2.5, label='Shaped Rota')
    ax2.plot(tn[:, 1], tn[:, 0], color='#ef4444', linewidth=2.0, linestyle='--', label='Naive Rota')
    ax2.scatter([0], [0], color='#f59e0b', s=160, marker='*', zorder=5)

    # Koridor sınırları
    y_range = np.linspace(0, 25, 100)
    ax2.fill_between(y_range, -y_range * np.tan(cone_angle), y_range * np.tan(cone_angle),
                     color='#38bdf8', alpha=0.10, label='LOS Emniyet Koridoru (±25°)')
    ax2.set_title("2. X-Y Düzlemi (Yerçekimi Gradyanı & Coriolis)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Along-Track Y (Metre) [İstasyona Mesafe]", color='#94a3b8')
    ax2.set_ylabel("Radyal Sapma X (Metre)", color='#94a3b8')
    ax2.invert_xaxis()
    ax2.legend(loc='lower left', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.25)

    # ----------------------------------------------------
    # PANEL 3: Z-Y Düzlemi (Cross-Track / Düzlem Dışı Salınım)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3, facecolor='#0d1424')
    ax3.plot(ts[:, 1], ts[:, 2], color='#10b981', linewidth=2.5, label='Shaped Düzlem Sönümlemesi')
    ax3.plot(tn[:, 1], tn[:, 2], color='#ef4444', linewidth=2.0, linestyle='--', label='Naive Düzlem Hareketi')
    ax3.scatter([0], [0], color='#f59e0b', s=160, marker='*', zorder=5)
    ax3.axhline(0, color='#64748b', linestyle=':', alpha=0.6)
    ax3.set_title("3. Z-Y Düzlemi (Düzlem Dışı Harmonik Salınım)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Along-Track Y (Metre)", color='#94a3b8')
    ax3.set_ylabel("Cross-Track Z Sapması (Metre)", color='#94a3b8')
    ax3.invert_xaxis()
    ax3.legend(loc='lower left', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax3.grid(True, linestyle=':', alpha=0.25)

    # ----------------------------------------------------
    # PANEL 4: Faz Portresi (Mesafe vs Bağıl Hız Koridoru)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4, facecolor='#0d1424')
    dist_s = np.linalg.norm(ts[:, :3], axis=1)
    speed_s = np.linalg.norm(ts[:, 3:6], axis=1)
    dist_n = np.linalg.norm(tn[:, :3], axis=1)
    speed_n = np.linalg.norm(tn[:, 3:6], axis=1)

    d_eval = np.linspace(0, 25, 200)
    v_safe_curve = 0.03 + 0.47 * np.tanh(d_eval / 8.0)

    ax4.plot(d_eval, v_safe_curve, color='#f59e0b', linewidth=2.0, linestyle='-.', label='Asimptotik Hız Koridoru (v_safe)')
    ax4.plot(dist_s, speed_s, color='#10b981', linewidth=2.5, label='Shaped Ajan Faz Eğrisi')
    ax4.plot(dist_n, speed_n, color='#ef4444', linewidth=2.0, linestyle='--', label='Naive Ajan (Hovering Düşüşü)')

    ax4.axvspan(0.0, 0.15, color='#10b981', alpha=0.2, label='Kenetlenme Bölgesi (<=0.15m)')
    ax4.axhline(0.06, color='#f43f5e', linestyle=':', label='Kaza Hız Eşiği (0.06 m/s)')
    ax4.set_title("4. Faz Portresi: Mesafe - Hız Yörüngesi", color='#38bdf8', fontsize=12, fontweight='bold')
    ax4.set_xlabel("Hedefe Kalan Mesafe (m)", color='#94a3b8')
    ax4.set_ylabel("Bağıl Hız (m/s)", color='#94a3b8')
    ax4.invert_xaxis()
    ax4.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax4.grid(True, linestyle=':', alpha=0.25)

    # ----------------------------------------------------
    # PANEL 5: 3 Eksenli RCS İtki Kuvvetleri Dağılımı (Fx, Fy, Fz)
    # ----------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5, facecolor='#0d1424')
    time_s = np.arange(len(fs))
    ax5.plot(time_s, fs[:, 0], color='#ef4444', linewidth=1.4, alpha=0.85, label='Fx (Radyal / Coriolis İtkisi)')
    ax5.plot(time_s, fs[:, 1], color='#10b981', linewidth=1.6, alpha=0.95, label='Fy (Along-Track Fren/İtiş)')
    ax5.plot(time_s, fs[:, 2], color='#38bdf8', linewidth=1.4, alpha=0.85, label='Fz (Cross-Track Denge)')
    ax5.axhline(0, color='#64748b', linestyle=':', alpha=0.5)
    ax5.set_title("5. 3-Eksenli RCS İtici Ateşleme Profili (Newton)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax5.set_xlabel("Zaman (Saniye)", color='#94a3b8')
    ax5.set_ylabel("İtici Kuvveti (N)", color='#94a3b8')
    ax5.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax5.grid(True, linestyle=':', alpha=0.25)

    # ----------------------------------------------------
    # PANEL 6: Anlık Adım Ödülü ve Potansiyel Havuzu
    # ----------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#0d1424')
    ax6.plot(np.arange(len(rewards_shaped)), rewards_shaped, color='#10b981', linewidth=2.0, label='Shaped (Üstel Teşvik Sinyali)')
    ax6.plot(np.arange(len(rewards_naive)), rewards_naive, color='#ef4444', linewidth=1.5, linestyle='--', label='Naive (Ceza Korkusu Sinyali)')
    ax6.set_title("6. Adım Başına RL Ödül Akışı (Reward Signal)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax6.set_xlabel("Adım (Step)", color='#94a3b8')
    ax6.set_ylabel("Ödül Değeri (Reward)", color='#94a3b8')
    ax6.legend(loc='lower right', facecolor='#1e293b', edgecolor='none', fontsize=8)
    ax6.grid(True, linestyle=':', alpha=0.25)

    # Çıktıyı kaydet
    out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbital_3d_deep_dive.png")
    plt.savefig(out_img, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[GÖRSEL BAŞARIYLA ÜRETİLDİ] -> {out_img}")


if __name__ == "__main__":
    generate_deep_dive_plots()
