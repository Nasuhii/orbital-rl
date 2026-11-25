"""
Orbital-RL: Gerçekçi 3D Astrodinamik Yörünge ve Görev Planlama Programı
====================================================================
Bu program, uzay araçlarının Alçak Dünya Yörüngesindeki (LEO) tam uçuş
profilini gerçekçi fiziksel astrodinamik modelleriyle simüle eder ve çizer:

Mühendislik Modelleri:
  1. 3 Boyutlu ECI (Earth-Centered Inertial) Koordinat Çerçevesi
  2. İki Cisim Yerçekimi + J2 Yerküre Basıklık (Oblateness) Pertürbasyonu
  3. ECEF (Earth-Centered Earth-Fixed) Koordinat Dönüşümü & Dünya Dönüşü (Yer İzi)
  4. 350 km -> 400 km Hohmann Transfer & Phasing Manevraları (Delta-V1, Delta-V2)
  5. Hedefe 10 km Kala Clohessy-Wiltshire (CW) LVLH Bağıl Yaklaşma Rotası
  6. 4 Panelli 300 DPI Yüksek Çözünürlüklü Mühendislik Çizimi
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Sabitler (WGS-84 / Standart Astrodinamik)
MU_EARTH = 3.986004418e14       # m^3 / s^2 (Dünya Standart Yerçekimi Parametresi)
R_EARTH  = 6378137.0            # metre (Dünya Ekvator Yarıçapı)
J2       = 1.08262668e-3        # J2 Zonel Harmonik (Yerküre basıklığı)
OMEGA_E  = 7.2921150e-5         # rad/s (Dünya Kendi Ekseni Dönüş Açısal Hızı)

# Görev Parametreleri
ALT_CHASER_PARK = 350_000.0     # 350 km Park Yörüngesi
ALT_TARGET_ISS  = 400_000.0     # 400 km Hedef (ISS) Yörüngesi
INCLINATION_DEG = 51.64         # ISS Yörünge Eğim Açısı (derece)
INCLINATION_RAD = np.radians(INCLINATION_DEG)


def j2_gravity_acceleration(r_vec):
    """
    J2 Basıklık Pertürbasyonu Eklenmiş Yerçekimi İvmesi (ECI Çerçevesi).
    r_vec: [x, y, z] metre
    """
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    r2 = r * r
    r5 = r2 * r2 * r
    z2 = z * z

    # Standart Newton İvmesi: -mu * r_vec / r^3
    a_newton = - (MU_EARTH / (r * r2)) * r_vec

    # J2 Pertürbasyon Bileşenleri
    factor = 1.5 * J2 * MU_EARTH * (R_EARTH ** 2) / r5
    z_term = 5.0 * z2 / r2

    a_j2_x = factor * x * (z_term - 1.0)
    a_j2_y = factor * y * (z_term - 1.0)
    a_j2_z = factor * z * (z_term - 3.0)

    a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
    return a_newton + a_j2


def rk4_orbit_step(state, dt):
    """
    4. Dereceden Runge-Kutta (RK4) Sayısal İntegratörü
    state: [x, y, z, vx, vy, vz]
    """
    r0 = state[:3]
    v0 = state[3:6]

    # k1
    k1_v = j2_gravity_acceleration(r0)
    k1_r = v0

    # k2
    r_half1 = r0 + 0.5 * dt * k1_r
    v_half1 = v0 + 0.5 * dt * k1_v
    k2_v = j2_gravity_acceleration(r_half1)
    k2_r = v_half1

    # k3
    r_half2 = r0 + 0.5 * dt * k2_r
    v_half2 = v0 + 0.5 * dt * k2_v
    k3_v = j2_gravity_acceleration(r_half2)
    k3_r = v_half2

    # k4
    r_full = r0 + dt * k3_r
    v_full = v0 + dt * k3_v
    k4_v = j2_gravity_acceleration(r_full)
    k4_r = v_full

    r_next = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return np.concatenate([r_next, v_next])


def eci_to_lat_lon(r_vec, t_sec):
    """
    ECI koordinatlarını Dünya dönüşü hesaba katılarak Enlem ve Boylama çevirir.
    """
    x, y, z = r_vec
    r_mag = np.linalg.norm(r_vec)
    
    # Greenwich Yıldız Zamanı Açısı (GST)
    theta_gst = OMEGA_E * t_sec

    # ECEF koordinatları (Z etrafında ters dönüş)
    x_ecef =  x * math.cos(theta_gst) + y * math.sin(theta_gst)
    y_ecef = -x * math.sin(theta_gst) + y * math.cos(theta_gst)
    z_ecef = z

    lat_rad = math.asin(np.clip(z_ecef / r_mag, -1.0, 1.0))
    lon_rad = math.atan2(y_ecef, x_ecef)

    return math.degrees(lat_rad), math.degrees(lon_rad)


def simulate_full_mission():
    """
    Uçtan Uca Yörünge Görevini Simüle Eder:
      1. Park Yörüngesi (1 Periyot)
      2. Hohmann Transferi (Delta-V1 ile 350 km -> 400 km)
      3. Daireselleştirme (Delta-V2 ile 400 km)
      4. ISS Yakınlaşma ve LVLH Bağıl Hareketi
    """
    print("\n" + "=" * 70)
    print("🛰️  GERÇEKÇİ ASTRODİNAMİK YÖRÜNGE VE GÖREV PLANLAMA SİMÜLASYONU")
    print("=" * 70)

    # 1. Yarı-büyük eksenler ve hızlar
    r1 = R_EARTH + ALT_CHASER_PARK   # 6,728,137 m
    r2 = R_EARTH + ALT_TARGET_ISS    # 6,778,137 m

    v1_circ = math.sqrt(MU_EARTH / r1)  # ~7697 m/s
    v2_circ = math.sqrt(MU_EARTH / r2)  # ~7668 m/s

    # Hohmann Transfer Elipsi
    a_tx = (r1 + r2) / 2.0
    v_tx_perigee = math.sqrt(MU_EARTH * (2.0 / r1 - 1.0 / a_tx))
    v_tx_apogee  = math.sqrt(MU_EARTH * (2.0 / r2 - 1.0 / a_tx))

    delta_v1 = v_tx_perigee - v1_circ  # ~14.1 m/s
    delta_v2 = v2_circ - v_tx_apogee   # ~14.1 m/s
    total_dv = delta_v1 + delta_v2

    t_hohmann = math.pi * math.sqrt((a_tx ** 3) / MU_EARTH) # ~2730 s (45.5 dk)
    t_period_1 = 2.0 * math.pi * math.sqrt((r1 ** 3) / MU_EARTH) # ~5480 s (91.3 dk)
    t_period_2 = 2.0 * math.pi * math.sqrt((r2 ** 3) / MU_EARTH) # ~5540 s (92.3 dk)

    print(f" • Kovalayıcı Park İrtifası    : {ALT_CHASER_PARK / 1000:.1f} km | Hız: {v1_circ:.2f} m/s | Periyot: {t_period_1/60:.1f} dk")
    print(f" • Hedef İstasyon (ISS) İrtifası: {ALT_TARGET_ISS / 1000:.1f} km | Hız: {v2_circ:.2f} m/s | Periyot: {t_period_2/60:.1f} dk")
    print(f" • Hohmann Ateşleme 1 (Delta-V1): +{delta_v1:.2f} m/s (Tırmanış Başlatma)")
    print(f" • Hohmann Ateşleme 2 (Delta-V2): +{delta_v2:.2f} m/s (400 km Daireselleştirme)")
    print(f" • Toplam Manevra Delta-V       : {total_dv:.2f} m/s | Transfer Süresi: {t_hohmann/60:.1f} dk")
    print("-" * 70)

    # Başlangıç Durumları (ECI Frame)
    # Eğim = 51.64 deg, RAAN = 0, Doğru Anomali = 0
    # Konum Ekvator düzleminden başlasın:
    # r = [r1 * cos(0), r1 * sin(0) * cos(i), r1 * sin(0) * sin(i)] = [r1, 0, 0]
    # v = [0, v1 * cos(i), v1 * sin(i)]
    r0_chaser = np.array([r1, 0.0, 0.0])
    v0_chaser = np.array([0.0, v1_circ * math.cos(INCLINATION_RAD), v1_circ * math.sin(INCLINATION_RAD)])
    state_chaser = np.concatenate([r0_chaser, v0_chaser])

    # Hedef (ISS) - Faz farkı ile 400 km yörüngede (Biraz ileride başlasın)
    theta_lead = math.radians(2.8) # Fazlama açısı
    r0_iss = np.array([
        r2 * math.cos(theta_lead),
        r2 * math.sin(theta_lead) * math.cos(INCLINATION_RAD),
        r2 * math.sin(theta_lead) * math.sin(INCLINATION_RAD)
    ])
    v0_iss = np.array([
        -v2_circ * math.sin(theta_lead),
        v2_circ * math.cos(theta_lead) * math.cos(INCLINATION_RAD),
        v2_circ * math.cos(theta_lead) * math.sin(INCLINATION_RAD)
    ])
    state_iss = np.concatenate([r0_iss, v0_iss])

    # Zaman Çizelgesi
    dt = 10.0 # 10 saniye adım büyüklüğü
    
    # 1. Faz: Park Yörüngesi (1/2 Periyot = ~2700 s)
    t_park = 2700.0
    
    # 2. Faz: Hohmann Transferi (~2730 s)
    
    # 3. Faz: Ortak Uçuş ve Yaklaşma (1/2 Periyot = ~2800 s)
    t_total = t_park + t_hohmann + 2800.0
    steps = int(t_total / dt)

    t_history = []
    chaser_pos = []
    chaser_vel = []
    iss_pos = []
    iss_vel = []
    phase_tags = []
    ground_lat = []
    ground_lon = []

    current_time = 0.0
    hohmann_boost1_done = False
    hohmann_boost2_done = False

    burn1_point = None
    burn2_point = None

    print("🚀 Sayısal Yörünge İntegrasyonu Başlatılıyor (RK4 + J2 Pertürbasyonu)...")

    for step in range(steps):
        t_history.append(current_time)
        chaser_pos.append(state_chaser[:3].copy())
        chaser_vel.append(state_chaser[3:6].copy())
        iss_pos.append(state_iss[:3].copy())
        iss_vel.append(state_iss[3:6].copy())

        lat, lon = eci_to_lat_lon(state_chaser[:3], current_time)
        ground_lat.append(lat)
        ground_lon.append(lon)

        # Manevra Kararları
        if current_time >= t_park and not hohmann_boost1_done:
            # 1. ATEŞLEME (Delta-V1): Hız vektörü doğrultusunda itki
            v_dir = state_chaser[3:6] / np.linalg.norm(state_chaser[3:6])
            state_chaser[3:6] += v_dir * delta_v1
            hohmann_boost1_done = True
            burn1_point = state_chaser[:3].copy()
            phase_tags.append("BURST-1 (Hohmann)")
        elif current_time >= (t_park + t_hohmann) and not hohmann_boost2_done:
            # 2. ATEŞLEME (Delta-V2): Daireselleştirme
            v_dir = state_chaser[3:6] / np.linalg.norm(state_chaser[3:6])
            state_chaser[3:6] += v_dir * delta_v2
            hohmann_boost2_done = True
            burn2_point = state_chaser[:3].copy()
            phase_tags.append("BURST-2 (Daireselleştirme)")
        elif not hohmann_boost1_done:
            phase_tags.append("Park Yörüngesi (350 km)")
        elif not hohmann_boost2_done:
            phase_tags.append("Hohmann Transferi")
        else:
            phase_tags.append("Kenetlenme Fazı (400 km)")

        # RK4 Adımları
        state_chaser = rk4_orbit_step(state_chaser, dt)
        state_iss = rk4_orbit_step(state_iss, dt)
        current_time += dt

    chaser_pos = np.array(chaser_pos)
    iss_pos = np.array(iss_pos)
    t_history = np.array(t_history)
    ground_lat = np.array(ground_lat)
    ground_lon = np.array(ground_lon)

    print("✅ Yörünge İntegrasyonu Tamamlandı! Grafikler Çiziliyor...")
    return {
        't': t_history,
        'chaser_pos': chaser_pos,
        'chaser_vel': np.array(chaser_vel),
        'iss_pos': iss_pos,
        'iss_vel': np.array(iss_vel),
        'lat': ground_lat,
        'lon': ground_lon,
        'burn1': burn1_point,
        'burn2': burn2_point,
        't_park': t_park,
        't_hohmann': t_hohmann,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'total_dv': total_dv
    }


def compute_relative_cw_trajectory(sim_data):
    """
    Hedefe son yaklaşma esnasında ISS merkezli LVLH koordinatlarını hesaplar.
    x_rel: Radyal (Yerçekimi gradyanı)
    y_rel: Along-Track (V-Bar yörünge hızı yönü)
    z_rel: Cross-Track (Açısal momentum / düzlem dışı)
    """
    rel_positions = []
    # Son 120 adımı (son 20 dakika) inceleyelim
    start_idx = max(0, len(sim_data['t']) - 120)

    for i in range(start_idx, len(sim_data['t'])):
        r_t = sim_data['iss_pos'][i]
        v_t = sim_data['iss_vel'][i]
        r_c = sim_data['chaser_pos'][i]

        # LVLH Birim Vektörleri
        # x_hat = r_t / |r_t| (Radyal dışa doğru)
        # z_hat = (r_t x v_t) / |r_t x v_t| (Açısal momentum)
        # y_hat = z_hat x x_hat (V-Bar / Along-track)
        r_norm = np.linalg.norm(r_t)
        x_hat = r_t / r_norm
        h_vec = np.cross(r_t, v_t)
        z_hat = h_vec / np.linalg.norm(h_vec)
        y_hat = np.cross(z_hat, x_hat)

        # Bağıl Vektör (ECI -> LVLH)
        dr = r_c - r_t
        x_rel = np.dot(dr, x_hat)
        y_rel = np.dot(dr, y_hat)
        z_rel = np.dot(dr, z_hat)

        rel_positions.append([x_rel, y_rel, z_rel])

    return np.array(rel_positions)


def plot_engineering_dashboard(sim_data, output_path):
    """
    4 Panelli Yüksek Çözünürlüklü Mühendislik Dashboard'u Üretir.
    """
    fig = plt.figure(figsize=(20, 14), facecolor='#090d16')
    plt.subplots_adjust(left=0.05, right=0.96, top=0.93, bottom=0.06, wspace=0.25, hspace=0.28)

    fig.suptitle(
        "🛰️ ORBITAL-RL: GERÇEKÇİ 3D YÖRÜNGE, HOHMANN TRANSFERİ VE YER İZİ PROGRAMI",
        fontsize=18, fontweight='bold', color='#38bdf8', y=0.98
    )

    gs = GridSpec(2, 2, figure=fig)

    # -------------------------------------------------------------
    # PANEL 1: 3D ECI Uzay Görünümü & Yörüngeler
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d', facecolor='#050811')
    ax1.set_title("1. 3D ECI Küresel Yörünge Görünümü (J2 Pertürbasyonu)", color='#f8fafc', fontsize=12, pad=10, fontweight='bold')

    # Dünya Küresi Çizimi
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = (R_EARTH / 1000.0) * np.outer(np.cos(u), np.sin(v))
    y_earth = (R_EARTH / 1000.0) * np.outer(np.sin(u), np.sin(v))
    z_earth = (R_EARTH / 1000.0) * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='#1e3a8a', alpha=0.35, edgecolor='#3b82f6', linewidth=0.2)

    # Ekvator Çemberi
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot((R_EARTH / 1000.0) * np.cos(theta), (R_EARTH / 1000.0) * np.sin(theta), np.zeros(100),
             color='#64748b', linestyle='--', linewidth=1.0, label='Ekvator Düzlemi')

    # Yörünge Yolları (km cinsinden)
    chaser_km = sim_data['chaser_pos'] / 1000.0
    iss_km = sim_data['iss_pos'] / 1000.0

    # Fazlara göre böl
    t_park = sim_data['t_park']
    t_hohmann = sim_data['t_hohmann']
    idx1 = int(t_park / (sim_data['t'][1] - sim_data['t'][0]))
    idx2 = int((t_park + t_hohmann) / (sim_data['t'][1] - sim_data['t'][0]))

    ax1.plot(chaser_km[:idx1, 0], chaser_km[:idx1, 1], chaser_km[:idx1, 2],
             color='#38bdf8', linewidth=2.0, label=f'Park Yörüngesi ({ALT_CHASER_PARK/1000:.0f} km)')
    ax1.plot(chaser_km[idx1:idx2, 0], chaser_km[idx1:idx2, 1], chaser_km[idx1:idx2, 2],
             color='#f59e0b', linewidth=2.5, linestyle='-', label='Hohmann Transfer Elipsi')
    ax1.plot(chaser_km[idx2:, 0], chaser_km[idx2:, 1], chaser_km[idx2:, 2],
             color='#10b981', linewidth=2.0, label=f'Hedef Yörüngesi ({ALT_TARGET_ISS/1000:.0f} km)')

    # ISS Yolu
    ax1.plot(iss_km[:, 0], iss_km[:, 1], iss_km[:, 2],
             color='#94a3b8', linewidth=1.0, linestyle=':', alpha=0.7, label='ISS Referans Rotası')

    # Ateşleme Noktaları
    if sim_data['burn1'] is not None:
        b1 = sim_data['burn1'] / 1000.0
        ax1.scatter(b1[0], b1[1], b1[2], color='#ef4444', s=70, marker='^', zorder=10,
                    label=f'ΔV1: +{sim_data["delta_v1"]:.1f} m/s')
    if sim_data['burn2'] is not None:
        b2 = sim_data['burn2'] / 1000.0
        ax1.scatter(b2[0], b2[1], b2[2], color='#a855f7', s=70, marker='v', zorder=10,
                    label=f'ΔV2: +{sim_data["delta_v2"]:.1f} m/s')

    ax1.set_xlabel('X [km]', color='#94a3b8')
    ax1.set_ylabel('Y [km]', color='#94a3b8')
    ax1.set_zlabel('Z [km]', color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    ax1.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#f8fafc')
    ax1.view_init(elev=28, azim=45)

    # -------------------------------------------------------------
    # PANEL 2: 2D Yer İzi (Ground Track - Mercator)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#050811')
    ax2.set_title("2. Dünya Yer İzi (Ground Track - Enlem / Boylam)", color='#f8fafc', fontsize=12, fontweight='bold')

    # Grid ve Ekvator çizgisi
    ax2.axhline(0, color='#64748b', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.axhline(INCLINATION_DEG, color='#f43f5e', linestyle=':', linewidth=1.0, alpha=0.8, label=f'+{INCLINATION_DEG}° Maksimum Enlem')
    ax2.axhline(-INCLINATION_DEG, color='#f43f5e', linestyle=':', linewidth=1.0, alpha=0.8, label=f'-{INCLINATION_DEG}° Minimum Enlem')

    # Boylam sarma (wrap-around) kesintilerini düzelt
    lon = sim_data['lon']
    lat = sim_data['lat']
    diffs = np.where(np.abs(np.diff(lon)) > 180.0)[0]
    segments = np.split(np.arange(len(lon)), diffs + 1)

    for seg in segments:
        ax2.plot(lon[seg], lat[seg], color='#38bdf8', linewidth=1.8)

    # Ateşleme noktalarını yer izinde göster
    if sim_data['burn1'] is not None:
        lat1, lon1 = eci_to_lat_lon(sim_data['burn1'], sim_data['t_park'])
        ax2.scatter(lon1, lat1, color='#ef4444', s=80, marker='^', zorder=10, label='ΔV1 Ateşleme Konumu')
    if sim_data['burn2'] is not None:
        lat2, lon2 = eci_to_lat_lon(sim_data['burn2'], sim_data['t_park'] + sim_data['t_hohmann'])
        ax2.scatter(lon2, lat2, color='#a855f7', s=80, marker='v', zorder=10, label='ΔV2 Ateşleme Konumu')

    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_xlabel('Boylam (Longitude) [Derece]', color='#94a3b8')
    ax2.set_ylabel('Enlem (Latitude) [Derece]', color='#94a3b8')
    ax2.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
    ax2.tick_params(colors='#94a3b8')
    ax2.legend(loc='lower left', facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#f8fafc')

    # -------------------------------------------------------------
    # PANEL 3: Uçuş Telemetrisi (İrtifa & Hız Zaman Serisi)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor='#050811')
    ax3.set_title("3. Görev İrtifa ve Orbital Hız Profili (Hohmann Geçişi)", color='#f8fafc', fontsize=12, fontweight='bold')

    t_min = sim_data['t'] / 60.0
    r_mag = np.linalg.norm(sim_data['chaser_pos'], axis=1)
    alt_km = (r_mag - R_EARTH) / 1000.0
    v_mag = np.linalg.norm(sim_data['chaser_vel'], axis=1)

    color_alt = '#38bdf8'
    ax3.plot(t_min, alt_km, color=color_alt, linewidth=2.2, label='İrtifa (km)')
    ax3.set_xlabel('Uçuş Süresi [Dakika]', color='#94a3b8')
    ax3.set_ylabel('İrtifa [km]', color=color_alt)
    ax3.tick_params(axis='y', labelcolor=color_alt)
    ax3.tick_params(axis='x', colors='#94a3b8')
    ax3.grid(True, color='#1e293b', linestyle='--', alpha=0.7)

    # İkiz Eksen: Hız
    ax3_twin = ax3.twinx()
    color_vel = '#f59e0b'
    ax3_twin.plot(t_min, v_mag, color=color_vel, linewidth=1.8, linestyle='-.', label='Orbital Hız (m/s)')
    ax3_twin.set_ylabel('Orbital Hız [m/s]', color=color_vel)
    ax3_twin.tick_params(axis='y', labelcolor=color_vel)

    # Ateşleme işaretleri
    ax3.axvline(sim_data['t_park'] / 60.0, color='#ef4444', linestyle=':', label='ΔV1 (Tırmanış)')
    ax3.axvline((sim_data['t_park'] + sim_data['t_hohmann']) / 60.0, color='#a855f7', linestyle=':', label='ΔV2 (Daireselleştirme)')

    ax3.legend(loc='center left', facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#f8fafc')
    ax3_twin.legend(loc='center right', facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#f8fafc')

    # -------------------------------------------------------------
    # PANEL 4: LVLH Hedef Merkezli Son Yaklaşma Rotası (Clohessy-Wiltshire)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor='#050811')
    ax4.set_title("4. LVLH Hedef Merkezli Bağıl Yaklaşma ve V-Bar Girişi", color='#f8fafc', fontsize=12, fontweight='bold')

    rel_pos = compute_relative_cw_trajectory(sim_data)
    # rel_pos[:, 0] = Radial (x), rel_pos[:, 1] = Along-track (y), rel_pos[:, 2] = Cross-track (z)
    
    # 2D Görünüm: Along-Track (Y) vs Radial (X)
    y_along = rel_pos[:, 1] / 1000.0  # km
    x_radial = rel_pos[:, 0] / 1000.0 # km

    ax4.plot(y_along, x_radial, color='#10b981', linewidth=2.2, label='Bağıl Yaklaşma Sarmalı')
    ax4.scatter(y_along[0], x_radial[0], color='#38bdf8', s=70, label='Yaklaşma Başlangıcı (-Y)')
    ax4.scatter(0, 0, color='#facc15', marker='*', s=180, zorder=10, label='🎯 ISS Kenetlenme Portu (0,0)')

    # V-Bar Kenetlenme Koridoru (25 derecelik yaklaşma konisi)
    cone_dist = np.linspace(0, max(0.5, np.max(np.abs(y_along)) * 0.3), 30)
    cone_w = cone_dist * np.tan(np.radians(25.0))
    ax4.fill_between(cone_dist, -cone_w, cone_w, color='#10b981', alpha=0.15, label='V-Bar Güvenli Koni (±25°)')

    ax4.set_xlabel('Along-Track / V-Bar (Y) [km]  [Geri <---> İleri]', color='#94a3b8')
    ax4.set_ylabel('Radial / Yerçekimi Gradyanı (X) [km]  [Aşağı <---> Yukarı]', color='#94a3b8')
    ax4.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
    ax4.tick_params(colors='#94a3b8')
    ax4.legend(loc='lower left', facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#f8fafc')

    # Bilgi Kutusu
    info_text = (
        f"GÖREV TELEMETRİSİ:\n"
        f"• Park İrtifası : {ALT_CHASER_PARK/1000:.0f} km\n"
        f"• Hedef İrtifası: {ALT_TARGET_ISS/1000:.0f} km\n"
        f"• ΔV1 (Hohmann) : +{sim_data['delta_v1']:.2f} m/s\n"
        f"• ΔV2 (Dairesel): +{sim_data['delta_v2']:.2f} m/s\n"
        f"• Toplam ΔV     : {sim_data['total_dv']:.2f} m/s\n"
        f"• Simülasyon    : RK4 + J2 Grav."
    )
    ax4.text(
        0.97, 0.96, info_text, transform=ax4.transAxes,
        fontsize=8.5, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.9),
        color='#f8fafc', fontfamily='monospace'
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 [MÜHENDİSLİK GRAFİĞİ KAYDEDİLDİ] -> {output_path}")


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realistic_orbital_trajectory.png")
    data = simulate_full_mission()
    plot_engineering_dashboard(data, out_file)
    print("\n🚀 Gerçekçi yörünge simülasyonu ve çizimi başarıyla tamamlandı!\n")
