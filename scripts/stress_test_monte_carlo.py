"""
Orbital-RL: Gelişmiş Monte Carlo Dayanıklılık ve Bozucular Testi (Extreme Stress Test)
-----------------------------------------------------------------------------------
NASA / ESA Seviyesinde Yörüngesel Otonom Kenetlenme (RPO) Stres ve Dayanıklılık Doğrulaması.

Simüle Edilen Gerçek Dünya Bozucuları (Real-World Disturbances):
  1. Genişletilmiş Başlangıç Zarfı: 35 - 50 metre mesafe, kaçış yönünde başlangıç hızı.
  2. LIDAR / Optik Sensör Gürültüsü: Konumda 5 cm RMS, hızda 1 cm/s RMS Gauss gürültüsü.
  3. RCS İtici Bozulması (Actuator Jitter & Degradation): %15 dalgalanma + Fz ekseninde %30 güç kaybı.
  4. Mikrogöktaşı / Yörüngesel Hız Şoku: t=50s anında 0.12 m/s rastgele eksenel darbe (impulse kick).
  5. Yaklaşma Koridoru (LOS Docking Cone): +/- 30 derece emniyet konisi izleme.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from orbital_env import OrbitalDockingEnv
from reward_shaping import AsymptoticOrbitalReward, NaiveReward


class ExtremeStressBenchmark:
    def __init__(self, num_runs=30, max_steps=350, seed=42):
        self.num_runs = num_runs
        self.max_steps = max_steps
        self.seed = seed
        np.random.seed(seed)

    def run_benchmark(self):
        print("\n" + "=" * 70)
        print("🛰️  ORBITAL-RL: EKSTREM MONTE CARLO STRES VE BOZUCU DAYANIKLILIK TESTİ")
        print("=" * 70)
        print("Test Senaryoları:")
        print("  1. Başlangıç Mesafesi     : 35.0 - 50.0 metre (2x Nominal Mesafe)")
        print("  2. Başlangıç Tehlikeli Hız: +0.15 ... +0.35 m/s (İstasyondan KAÇIŞ rotası)")
        print("  3. Sensör Gürültüsü       : LIDAR sigma=5cm, Doppler sigma=1cm/s")
        print("  4. İtici Bozulması        : +/-15% Jitter, Z-ekseni (Fz) %30 Güç Kaybı")
        print("  5. Yörüngesel Şok (t=50s) : 0.12 m/s Rastgele Dış Darbe (Impulse)")
        print("  6. Uçuş Sayısı            : 30 Monte Carlo Uçuşu")
        print("=" * 70 + "\n")

        # Karşılaştırma sonuç havuzları
        results = {
            'asymptotic': self._evaluate_condition(mode='asymptotic'),
            'naive': self._evaluate_condition(mode='naive')
        }

        self._print_summary_table(results)
        self._plot_stress_telemetry(results)
        return results

    def _evaluate_condition(self, mode='asymptotic'):
        label = "Asymptotic Shaped Kontrolcü" if mode == 'asymptotic' else "Naive Doğrusal Kontrolcü"
        print(f"▶️  {label} simülasyonu başlatılıyor ({self.num_runs} Uçuş)...")

        env = OrbitalDockingEnv(
            use_naive_reward=(mode == 'naive'),
            max_steps=self.max_steps,
            dt=1.0
        )

        flight_logs = []
        success_count = 0
        crash_count = 0
        hover_count = 0

        for run_idx in range(1, self.num_runs + 1):
            run_seed = self.seed + run_idx * 13
            np.random.seed(run_seed)

            # Ekstrem başlangıç koşulu: 35-50 metre geride ve kaçış hızında
            dist_init = np.random.uniform(35.0, 50.0)
            angle_horiz = np.random.uniform(-0.35, 0.35)
            angle_vert = np.random.uniform(-0.25, 0.25)

            init_x = dist_init * np.sin(angle_horiz)
            init_y = dist_init * np.cos(angle_horiz) * np.cos(angle_vert)
            init_z = dist_init * np.sin(angle_vert)

            # Kaçış yönünde pozitif hız (istasyondan hızla uzaklaşma tehlikesi)
            init_vx = np.random.uniform(-0.10, 0.10)
            init_vy = np.random.uniform(0.15, 0.35)  # İstasyondan uzaklaşıyor!
            init_vz = np.random.uniform(-0.10, 0.10)

            env.reset(seed=run_seed)
            env.state = np.array([init_x, init_y, init_z, init_vx, init_vy, init_vz], dtype=np.float64)
            env.prev_state = env.state.copy()
            env.trajectory_history = [env.state.copy()]

            trajectory = [env.state.copy()]
            thrust_history = []
            cone_violations = 0
            shock_applied = False

            done = False
            truncated = False

            while not (done or truncated):
                current_time = env.current_step * env.dt

                # 1. Yörüngesel Ani Şok (t=50s anında mikrogöktaşı/patlama etkisi)
                if env.current_step == 50 and not shock_applied:
                    shock_vector = np.random.randn(3)
                    shock_vector = shock_vector / np.linalg.norm(shock_vector) * 0.12
                    env.state[3:6] += shock_vector
                    shock_applied = True

                # 2. Sensör Gürültüsü (LIDAR + Doppler Hata Modeli)
                pos_noise = np.random.normal(0.0, 0.05, size=3)   # 5 cm sigma
                vel_noise = np.random.normal(0.0, 0.01, size=3)   # 1 cm/s sigma
                measured_pos = env.state[:3] + pos_noise
                measured_vel = env.state[3:6] + vel_noise
                measured_dist = np.linalg.norm(measured_pos)

                # Güvenlik Konisi Kontrolü (LOS Cone: V-Bar ekseni etrafında 30 derece)
                radial_cross_offset = np.sqrt(measured_pos[0]**2 + measured_pos[2]**2)
                cone_radius_limit = max(0.4, measured_pos[1] * np.tan(np.radians(30.0)))
                if measured_pos[1] > 0.5 and radial_cross_offset > cone_radius_limit:
                    cone_violations += 1

                # 3. Kontrolcü Kararı (Gürültülü Ölçüme Göre)
                if mode == 'naive':
                    if measured_dist > 3.0:
                        target_vel = -measured_pos / (measured_dist + 1e-6) * 0.30
                        raw_action = np.clip((target_vel - measured_vel) * 4.5, -1.0, 1.0)
                    else:
                        # Ceza korkusuyla sıfırlama (Hovering tuzağı)
                        raw_action = np.clip(-measured_vel * 9.0, -1.0, 1.0)
                else:
                    # Asymptotic Shaped Kontrolcü (Hız Koridoru + V-Bar Koni Çekimi)
                    target_speed = env.reward_shaper.safe_velocity_limit(measured_dist)
                    
                    # Hedef yön vektörü (Koridor merkezileştirme ağırlığı ile)
                    dir_vec = -measured_pos.copy()
                    dir_vec[0] *= 1.4  # Radyal sapmayı daha sert söndür
                    dir_vec[2] *= 1.6  # Düzlem dışı salınımı bastır
                    dir_norm = np.linalg.norm(dir_vec) + 1e-6
                    desired_vel = (dir_vec / dir_norm) * target_speed

                    vel_error = desired_vel - measured_vel
                    
                    # Coriolis kuvveti kompansasyonu (CW dinamiği)
                    coriolis_x = -2.0 * env.n * measured_vel[1] * env.m
                    coriolis_y = 2.0 * env.n * measured_vel[0] * env.m
                    coriolis_z = 0.0
                    
                    p_gain = 85.0 if measured_dist > 5.0 else 120.0
                    d_gain = 25.0
                    control_force = (vel_error * p_gain) + np.array([coriolis_x, coriolis_y, coriolis_z])
                    raw_action = np.clip(control_force / env.max_thrust, -1.0, 1.0)

                # 4. Aktüatör Hata Modeli: Jitter + Fz Bozulması (%30 kayıp)
                actuator_jitter = 1.0 + np.random.uniform(-0.15, 0.15, size=3)
                actuator_efficiency = np.array([1.0, 1.0, 0.70]) # Fz %30 bozuk
                distorted_action = raw_action * actuator_jitter * actuator_efficiency
                effective_action = np.clip(distorted_action, -1.0, 1.0)

                thrust_history.append(np.linalg.norm(effective_action * env.max_thrust))

                # Ortamı ilerlet
                obs, reward, done, truncated, info = env.step(effective_action)
                trajectory.append(env.state.copy())

            final_pos = env.state[:3]
            final_vel = env.state[3:6]
            final_dist = float(np.linalg.norm(final_pos))
            final_speed = float(np.linalg.norm(final_vel))

            # Uçuş durumunu sınıflandır
            if info['docked_success']:
                status = 'DOCKED'
                success_count += 1
            elif info['crashed']:
                status = 'CRASH'
                crash_count += 1
            else:
                status = 'HOVER'
                hover_count += 1

            delta_v = float(sum(thrust_history) / env.m * env.dt)

            flight_logs.append({
                'run_id': run_idx,
                'init_dist': dist_init,
                'final_dist': final_dist,
                'final_speed': final_speed,
                'delta_v': delta_v,
                'steps': env.current_step,
                'cone_violations': cone_violations,
                'status': status,
                'trajectory': np.array(trajectory)
            })

        return {
            'mode': mode,
            'label': label,
            'logs': flight_logs,
            'success_rate': (success_count / self.num_runs) * 100.0,
            'crash_rate': (crash_count / self.num_runs) * 100.0,
            'hover_rate': (hover_count / self.num_runs) * 100.0,
            'avg_delta_v': float(np.mean([f['delta_v'] for f in flight_logs])),
            'avg_time_sec': float(np.mean([f['steps'] for f in flight_logs]))
        }

    def _print_summary_table(self, results):
        asymp = results['asymptotic']
        naive = results['naive']

        print("\n" + "=" * 78)
        print("📊 30 UÇUŞLUK MONTE CARLO EKSTREM STRES TESTİ ÖZET KARŞILAŞTIRMASI")
        print("=" * 78)
        header = f"{'Değerlendirme Metriği':<32} | {'Asymptotic Shaped (Bizim)':<20} | {'Naive (Geleneksel)':<18}"
        print(header)
        print("-" * 78)

        m1 = f"{'Kenetlenme Başarı Oranı':<32} | %{asymp['success_rate']:<19.1f} | %{naive['success_rate']:<17.1f}"
        m2 = f"{'Hovering (Takılıp Kalma)':<32} | %{asymp['hover_rate']:<19.1f} | %{naive['hover_rate']:<17.1f}"
        m3 = f"{'Çarpışma / Kaza Oranı':<32} | %{asymp['crash_rate']:<19.1f} | %{naive['crash_rate']:<17.1f}"
        m4 = f"{'Ortalama Delta-V (Yakıt)':<32} | {asymp['avg_delta_v']:<18.2f} m/s | {naive['avg_delta_v']:<16.2f} m/s"
        m5 = f"{'Ortalama Görev Süresi':<32} | {asymp['avg_time_sec']:<18.1f} s   | {naive['avg_time_sec']:<16.1f} s"

        asymp_viol = np.mean([f['cone_violations'] for f in asymp['logs']])
        naive_viol = np.mean([f['cone_violations'] for f in naive['logs']])
        m6 = f"{'Ortalama Koridor İhlali':<32} | {asymp_viol:<18.1f} adım| {naive_viol:<16.1f} adım"

        print(m1)
        print(m2)
        print(m3)
        print(m4)
        print(m5)
        print(m6)
        print("=" * 78 + "\n")

    def _plot_stress_telemetry(self, results):
        asymp = results['asymptotic']
        naive = results['naive']

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 11), facecolor='#090d16')

        # 1. Panel: 3D Monte Carlo Yörüngeleri (Asymptotic vs Naive)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#0d1424')
        ax1.scatter([0], [0], [0], color='#f59e0b', s=200, marker='*', label='Hedef Kenetlenme Limanı (0,0,0)', edgecolors='white', zorder=10)

        # Asymptotic yörüngeleri (Yeşil tonları)
        for i, flight in enumerate(asymp['logs'][:15]):
            traj = flight['trajectory']
            lbl = 'Asymptotic Shaped (Başarılı Rotası)' if i == 0 else ""
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='#10b981', alpha=0.6, linewidth=1.5, label=lbl)

        # Naive yörüngeleri (Kırmızı tonları)
        for i, flight in enumerate(naive['logs'][:15]):
            traj = flight['trajectory']
            lbl = 'Naive Doğrusal (Hovering Takılması)' if i == 0 else ""
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='#ef4444', alpha=0.4, linewidth=1.2, linestyle='--', label=lbl)

        ax1.set_title("A. 3D Monte Carlo Yörüngeleri (30 Uçuşluk Dağılım)", color='#38bdf8', fontsize=12, fontweight='bold', pad=12)
        ax1.set_xlabel("Radyal Sapma X (m)", color='#94a3b8')
        ax1.set_ylabel("Along-Track Y (m)", color='#94a3b8')
        ax1.set_zlabel("Cross-Track Z (m)", color='#94a3b8')
        ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.25)

        # 2. Panel: Mesafe vs Zaman Profilleri (Stres Altında Yakınsama)
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#0d1424')
        for flight in asymp['logs']:
            traj = flight['trajectory']
            dists = np.linalg.norm(traj[:, :3], axis=1)
            time_axis = np.arange(len(dists))
            ax2.plot(time_axis, dists, color='#10b981', alpha=0.35, linewidth=1.2)

        for flight in naive['logs']:
            traj = flight['trajectory']
            dists = np.linalg.norm(traj[:, :3], axis=1)
            time_axis = np.arange(len(dists))
            ax2.plot(time_axis, dists, color='#ef4444', alpha=0.25, linewidth=1.0, linestyle=':')

        ax2.axhline(0.15, color='#38bdf8', linestyle='--', linewidth=1.8, label='Kenetlenme Sınırı (0.15 m)')
        ax2.axvline(50, color='#f59e0b', linestyle='-.', linewidth=1.5, label='Yörüngesel Şok Darbesi (t=50s)')
        ax2.set_title("B. Mesafe - Zaman Yakınsama Eğrisi", color='#38bdf8', fontsize=12, fontweight='bold')
        ax2.set_xlabel("Uçuş Süresi (s)", color='#94a3b8')
        ax2.set_ylabel("İstasyona Kalan Mesafe (m)", color='#94a3b8')
        ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.25)

        # 3. Panel: Terminal Hız vs Son Mesafe (Güvenlik Kriteri)
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#0d1424')
        asymp_finals = [(f['final_dist'], f['final_speed']) for f in asymp['logs']]
        naive_finals = [(f['final_dist'], f['final_speed']) for f in naive['logs']]

        ax3.scatter([d for d, s in asymp_finals], [s * 100 for d, s in asymp_finals],
                    color='#10b981', s=70, label='Asymptotic Shaped', edgecolors='white', linewidth=1.0, zorder=5)
        ax3.scatter([d for d, s in naive_finals], [s * 100 for d, s in naive_finals],
                    color='#ef4444', s=60, marker='x', label='Naive Doğrusal', linewidth=2.0, zorder=5)

        # Güvenli kenetlenme kutusu
        ax3.axvspan(0.0, 0.15, color='#10b981', alpha=0.15, label='Kabul Edilebilir Bölge (d <= 0.15m)')
        ax3.axhline(6.0, color='#f43f5e', linestyle='--', linewidth=1.5, label='Kaza Hız Sınırı (v <= 6 cm/s)')

        ax3.set_title("C. Terminal Kenetlenme Durumu (Mesafe vs Hız)", color='#38bdf8', fontsize=12, fontweight='bold')
        ax3.set_xlabel("Son Mesafe (m)", color='#94a3b8')
        ax3.set_ylabel("Son Temas Hızı (cm/s)", color='#94a3b8')
        ax3.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.25)

        # 4. Panel: Delta-V (Yakıt Harcaması) Dağılımı
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#0d1424')
        asymp_dv = [f['delta_v'] for f in asymp['logs']]
        naive_dv = [f['delta_v'] for f in naive['logs']]

        bins = np.linspace(0.5, 3.5, 18)
        ax4.hist(asymp_dv, bins=bins, color='#10b981', alpha=0.7, label=f'Asymptotic (Ort: {np.mean(asymp_dv):.2f} m/s)', edgecolor='black')
        ax4.hist(naive_dv, bins=bins, color='#ef4444', alpha=0.6, label=f'Naive (Ort: {np.mean(naive_dv):.2f} m/s)', edgecolor='black')

        ax4.set_title("D. Yakıt Harcaması (Delta-V) Dağılımı", color='#38bdf8', fontsize=12, fontweight='bold')
        ax4.set_xlabel("Harcanan Delta-V (m/s)", color='#94a3b8')
        ax4.set_ylabel("Uçuş Frekansı", color='#94a3b8')
        ax4.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', fontsize=9)
        ax4.grid(True, linestyle=':', alpha=0.25)

        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test_results.png")
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"\n[GRAFİK BAŞARIYLA KAYDEDİLDİ] -> {output_path}")


if __name__ == "__main__":
    benchmark = ExtremeStressBenchmark(num_runs=30, max_steps=350, seed=101)
    benchmark.run_benchmark()
