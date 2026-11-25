"""
Orbital-RL: 4K Yönetici ve Gelişim Sunumu Görselleştirici
---------------------------------------------------------
"Bu kadar geldik diye sunucam" talebi doğrultusunda projenin
0% Naive başlangıcından, Hovering kırılımına, MuJoCo temas fiziğine,
ve Şampiyon Modelin %100 başarısına kadar olan tüm mühendislik
yolculuğunu tek bir 4K Ultra-HD (3840x2160) sunum panosunda birleştirir.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(DIR_PATH, "orbital_rl_executive_presentation_showcase.png")

# Stil Ayarları
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2

fig = plt.figure(figsize=(25.6, 14.4), dpi=150, facecolor='#030712') # 3840 x 2160 pixels
gs = gridspec.GridSpec(100, 100, figure=fig, left=0.03, right=0.97, top=0.96, bottom=0.03)

# Renk Paleti
C_CYAN = '#38bdf8'
C_EMERALD = '#10b981'
C_AMBER = '#f59e0b'
C_ROSE = '#ef4444'
C_PURPLE = '#a855f7'
C_DARK_BG = '#090d16'
C_PANEL_BORDER = '#1e293b'

# ==============================================================================
# 1. ÜST BAŞLIK VE YÖNETİCİ ÖZETİ (HERO BANNER)
# ==============================================================================
ax_hero = fig.add_subplot(gs[0:12, :], facecolor='#080e1d')
ax_hero.set_xticks([])
ax_hero.set_yticks([])
for spine in ax_hero.spines.values():
    spine.set_color(C_CYAN)
    spine.set_linewidth(1.5)

ax_hero.text(0.015, 0.68, "ORBITAL-RL: OTONOM UZAY ARACI KENETLENMESİ VE GÜVENLİ DERİN RL GELİŞİM SUNUMU",
             fontsize=19, fontweight='bold', color='#ffffff', va='center')
ax_hero.text(0.015, 0.28, "0'dan Şampiyon Modele: 400 km LEO ISS Yörüngesinde Hovering Kırılımı, MuJoCo Fiziği ve %100 Başarılı Kenetlenme Yolculuğu",
             fontsize=11.5, color=C_CYAN, va='center')

# KPI Rozetleri
kpis = [
    ("🏆 SAF PPO (RAW)", "%100.0", C_EMERALD),
    ("🛡️ PPO + CBF + PWPF", "%82.0", C_EMERALD),
    ("🛑 KAZA / ÇARPIŞMA", "%0.0", C_EMERALD),
    ("🔒 TEMAS HIZI", "3.03 cm/s", C_EMERALD),
    ("⛽ DELTA-V (RAW/CBF)", "0.62 / 1.12", C_CYAN),
    ("🌐 DİJİTAL İKİZ", "4-CAM HD", C_AMBER)
]

for idx, (title, val, col) in enumerate(kpis):
    x_pos = 0.53 + idx * 0.077
    ax_hero.text(x_pos, 0.72, title, fontsize=7.5, fontweight='bold', color='#94a3b8', ha='center')
    ax_hero.text(x_pos, 0.30, val, fontsize=13, fontweight='bold', color=col, ha='center')


# ==============================================================================
# 2. GELİŞİM YOLCULUĞU: 4 TEMEL MÜHENDİSLİK SÜTUNU (ORTA BÖLÜM)
# ==============================================================================

# Sütun 1: BAŞLANGIÇ ÇIKMAZI (Hovering Patolojisi)
ax_col1 = fig.add_subplot(gs[15:58, 0:24], facecolor=C_DARK_BG)
ax_col1.set_title("FAZ 1 & 2: Başlangıç Çıkmazı & Hovering", fontsize=11.5, fontweight='bold', color=C_ROSE, pad=10)

# Simüle edilmiş Hovering Donma Eğrisi
d_pts = np.linspace(25, 0, 100)
# Naive PPO: 8m'de durur
v_naive = np.where(d_pts > 8.5, 0.35 * (d_pts - 8.5)/16.5, 0.0)
ax_col1.plot(d_pts, v_naive, color=C_ROSE, linewidth=2.2, label="Naive PPO (Donma / Hovering)")
ax_col1.axvline(x=8.5, color='#ef4444', linestyle=':', label="Hovering Kilidi (d = 8.5m)")
ax_col1.axhline(y=0.06, color='#64748b', linestyle='--', label="Kenetlenme Limiti (6 cm/s)")
ax_col1.set_xlabel("Mesafe (m)", fontsize=8.5, color='#94a3b8')
ax_col1.set_ylabel("Hız (m/s)", fontsize=8.5, color='#94a3b8')
ax_col1.set_xlim(25, 0) # Geriye doğru limana yaklaşma
ax_col1.set_ylim(-0.02, 0.45)
ax_col1.legend(loc='upper right', fontsize=7.0, framealpha=0.3)
ax_col1.grid(True)

desc_1 = (
    "• Sorun: -100 Crash cezasından korkan ajan limana 8m kala durur.\n"
    "• Başarı: %0.0 | Hovering: %100.0 | Ortalama Mesafe: 27.9m\n"
    "• Teşhis: Klasik doğrusal ödül (r = -dist) yerel minimuma düşer."
)
ax_col1.text(0.04, 0.22, desc_1, transform=ax_col1.transAxes, fontsize=8.0,
            color='#cbd5e1', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1b4b', edgecolor=C_ROSE, alpha=0.8))


# Sütun 2: KRİTİK KIRILMA (Asimptotik Vali & PBRS)
ax_col2 = fig.add_subplot(gs[15:58, 25:49], facecolor=C_DARK_BG)
ax_col2.set_title("FAZ 3: Asimptotik Hız Valisi & PBRS", fontsize=11.5, fontweight='bold', color=C_CYAN, pad=10)

v_safe_curve = np.minimum(0.40, 0.05 + 0.35 * np.tanh((d_pts - 1.0)/4.5))
ax_col2.fill_between(d_pts, v_safe_curve, 0.50, color=C_ROSE, alpha=0.15, label="Tehlikeli Bölge")
ax_col2.fill_between(d_pts, 0, v_safe_curve, color=C_EMERALD, alpha=0.12, label="Güvenli Set C")
ax_col2.plot(d_pts, v_safe_curve, color=C_CYAN, linewidth=2.2, label="v_safe(d) Asimptotik Vali")
ax_col2.axhline(y=0.06, color=C_ROSE, linestyle='--', label="Temas Limiti (6 cm/s)")
ax_col2.set_xlabel("Mesafe (m)", fontsize=8.5, color='#94a3b8')
ax_col2.set_ylabel("Hız (m/s)", fontsize=8.5, color='#94a3b8')
ax_col2.set_xlim(25, 0)
ax_col2.set_ylim(-0.02, 0.50)
ax_col2.legend(loc='upper right', fontsize=7.0, framealpha=0.3)
ax_col2.grid(True)

desc_2 = (
    "• Üstel Çekim: r_prox = 25*exp(-d/3.0) ajanı içeri çeker.\n"
    "• Asimptotik Vali: v_safe = 0.05 + 0.35*tanh((d-1)/4.5)\n"
    "• PBRS Teoremi: Durmanın negatif ödül ürettiği kanıtlandı."
)
ax_col2.text(0.04, 0.22, desc_2, transform=ax_col2.transAxes, fontsize=8.0,
            color='#cbd5e1', bbox=dict(boxstyle='round,pad=0.5', facecolor='#082f49', edgecolor=C_CYAN, alpha=0.8))


# Sütun 3: MUJOCO FİZİK GERÇEKLİĞİ & DOMAIN RANDOMIZATION
ax_col3 = fig.add_subplot(gs[15:58, 50:74], facecolor=C_DARK_BG)
ax_col3.set_title("FAZ 4: MuJoCo Fiziği & Güvenlik Kalkanı", fontsize=11.5, fontweight='bold', color=C_PURPLE, pad=10)

# Eyleyici PWPF Darbe Modülatörü
t_sim = np.linspace(0, 10, 200)
u_analog = 12.0 * np.sin(0.6 * t_sim)
u_pwpf = np.zeros_like(u_analog)
for i, u in enumerate(u_analog):
    if abs(u) > 4.5:
        u_pwpf[i] = 15.0 if u > 0 and (i % 6 < 3) else (-15.0 if u < 0 and (i % 6 < 3) else 0.0)

ax_col3.plot(t_sim, u_analog, color=C_CYAN, linewidth=1.6, label="Sinir Ağı Talebi (u_net)")
ax_col3.step(t_sim, u_pwpf, color=C_EMERALD, linewidth=1.4, alpha=0.9, label="PWPF RCS Darbeleri (±15N)")
ax_col3.axhline(y=15, color='#64748b', linestyle=':')
ax_col3.axhline(y=-15, color='#64748b', linestyle=':')
ax_col3.set_xlabel("Zaman (s)", fontsize=8.5, color='#94a3b8')
ax_col3.set_ylabel("İtki (N)", fontsize=8.5, color='#94a3b8')
ax_col3.set_ylim(-19, 19)
ax_col3.legend(loc='upper right', fontsize=7.0, framealpha=0.3)
ax_col3.grid(True)

desc_3 = (
    "• Rijit Gövde Dinamiği: MuJoCo temas ve sürtünme fiziği.\n"
    "• Domain Randomization: Kütle (400-600kg), İtki (12-18N).\n"
    "• CBF-QP Kalkanı: min ||u - u_rl||^2 s.t. h_dot + γh >= 0"
)
ax_col3.text(0.04, 0.22, desc_3, transform=ax_col3.transAxes, fontsize=8.0,
            color='#cbd5e1', bbox=dict(boxstyle='round,pad=0.5', facecolor='#3b0764', edgecolor=C_PURPLE, alpha=0.8))


# Sütun 4: ŞAMPİYON MODELİN DOĞRULANMASI (BENCHMARK)
ax_col4 = fig.add_subplot(gs[15:58, 75:100], facecolor=C_DARK_BG)
ax_col4.set_title("FAZ 5: Monte Carlo Doğrulama & Zirve", fontsize=11.5, fontweight='bold', color=C_EMERALD, pad=10)

bench_models = ["Naive", "Asymp", "MuJoCo DR", "BC+PPO", "700K CBF+PWPF", "700K Raw", "Safe-GNC"]
succ_vals    = [0, 0, 0, 0, 82, 100, 100]
hover_vals   = [100, 100, 43, 100, 18, 0, 0]
crash_vals   = [0, 0, 57, 0, 0, 0, 0]

idx = np.arange(len(bench_models))
w = 0.26
ax_col4.bar(idx - w, succ_vals, w, color=C_EMERALD, label="Başarı %")
ax_col4.bar(idx, hover_vals, w, color=C_CYAN, label="Hover %")
ax_col4.bar(idx + w, crash_vals, w, color=C_ROSE, label="Kaza %")

ax_col4.set_xticks(idx)
ax_col4.set_xticklabels(bench_models, rotation=30, ha='right', fontsize=7.5, color='#f8fafc')
ax_col4.set_ylabel("Oran (%)", fontsize=8.5, color='#94a3b8')
ax_col4.set_ylim(0, 115)
ax_col4.legend(loc='upper right', fontsize=7.0, framealpha=0.3)
ax_col4.grid(True, axis='y')

desc_4 = (
    "• 700K Champion-PPO: %100 (Raw) & %82 (CBF+PWPF)!\n"
    "• Sıfır Kaza (%0.0) | Analitik Servo Devralması Yok\n"
    "• Delta-V: 0.615 m/s (Raw) / 1.125 m/s (CBF+PWPF)."
)
ax_col4.text(0.04, 0.42, desc_4, transform=ax_col4.transAxes, fontsize=8.0,
            color='#cbd5e1', bbox=dict(boxstyle='round,pad=0.5', facecolor='#064e3b', edgecolor=C_EMERALD, alpha=0.8))


# ==============================================================================
# 3. ALT BÖLÜM: 4 KAMERALI CANLI MUJOCO GÖRSEL KANITLARI (VİDEO KARELERİ)
# ==============================================================================
cam_files = [
    ("chase", "1. Chase Kamerası (Takip & Yaklaşma)", "cam_chase_cam.png", 0, 24),
    ("probe", "2. FPV Prob HUD (Optik Nişangah)", "cam_probe_cam.png", 25, 49),
    ("cctv", "3. İstasyon CCTV (Liman Güvenlik)", "cam_station_cctv.png", 50, 74),
    ("tactical", "4. Yörünge Taktik (Geniş Açı LEO)", "cam_orbital_tactical.png", 75, 100)
]

for key, title, fname, c_start, c_end in cam_files:
    ax_cam = fig.add_subplot(gs[62:95, c_start:c_end], facecolor='#000000')
    ax_cam.set_title(title, fontsize=10.5, fontweight='bold', color=C_CYAN, pad=8)
    img_path = os.path.join(DIR_PATH, fname)
    if os.path.exists(img_path):
        try:
            im = Image.open(img_path)
            ax_cam.imshow(im)
        except Exception:
            ax_cam.text(0.5, 0.5, f"{key.upper()} FEED", color='#64748b', ha='center', va='center')
    else:
        ax_cam.text(0.5, 0.5, f"{key.upper()} FEED", color='#64748b', ha='center', va='center')
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_color(C_PANEL_BORDER)
        spine.set_linewidth(1.0)


# ==============================================================================
# 4. EN ALT BİLGİ VE ONAY ÇUBUĞU (FOOTER)
# ==============================================================================
ax_footer = fig.add_subplot(gs[96:100, :], facecolor='#080e1d')
ax_footer.set_xticks([])
ax_footer.set_yticks([])
for spine in ax_footer.spines.values():
    spine.set_color('#1e293b')

footer_text = (
    "🛰️ Proje: ORBITAL-RL (Deep RL + Safe GNC)  |  "
    "Simülasyon: MuJoCo Multi-Body Physics + Clohessy-Wiltshire qfrc  |  "
    "Canlı Görev Kontrol: http://localhost:8055/ (25 Hz Telemetri)  |  "
    "Tarih: 16.11.2026  |  Durum: 700K SAF PPO DOĞRULANDI (%100 RAW / %82 CBF+PWPF)"
)
ax_footer.text(0.5, 0.5, footer_text, fontsize=9.0, color='#94a3b8', ha='center', va='center', fontfamily='monospace')

# Kaydet
plt.savefig(OUTPUT_PNG, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ 4K Yönetici ve Gelişim Sunumu Başarıyla Üretildi: {OUTPUT_PNG}")
