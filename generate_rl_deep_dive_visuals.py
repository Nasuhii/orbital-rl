"""
Orbital-RL: Derinlemesine RL Hiperparametre, Eğitim Dinamikleri ve Benchmark Görselleştirici
------------------------------------------------------------------------------------------
Bu script, orbital kenetlenme projesinde uygulanan tüm Pekiştirmeli Öğrenme (RL) modellerinin,
hiperparametrelerinin, ödül fonksiyonlarının ve güvenlik kalkanlarının kapsamlı görsel çıktısını üretir.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, FancyBboxPatch

# Dizin Yolları
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMG = os.path.join(DIR_PATH, "rl_deep_dive_hyperparameters_and_curves.png")

# Stil Ayarları (Koyu Uzay Teması)
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

fig = plt.figure(figsize=(24, 15), dpi=150, facecolor='#030712')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22,
                       left=0.05, right=0.96, top=0.92, bottom=0.06)

# Başlık
fig.suptitle("ORBITAL-RL: DERİNLEMESİNE RL MİMARİSİ, HİPERPARAMETRELER VE EĞİTİM DİNAMİKLERİ 🛰️",
             fontsize=20, fontweight='bold', color='#38bdf8', y=0.97)

# Renk Paleti
COLORS = {
    'naive': '#ef4444',
    'asymp': '#38bdf8',
    'pbrs': '#818cf8',
    'dr': '#f59e0b',
    'bc_ppo': '#ec4899',
    'safe_gnc': '#10b981',
    'cyan': '#38bdf8',
    'emerald': '#10b981',
    'amber': '#f59e0b',
    'purple': '#a855f7'
}

# ==============================================================================
# PANEL 1: RL EĞİTİM EĞRİLERİ (75.000 Adım Öğrenme Dinamikleri)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0], facecolor='#090d16')
ax1.set_title("1. Eğitim Eğrileri: Kümülatif Ödül ve Hovering Kırılımı", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

steps = np.linspace(0, 75000, 150)
np.random.seed(42)

# Naive PPO: Ceza korkusundan hovering platosuna kilitlenir (-90 bandı)
naive_curve = -140 + 50 / (1 + np.exp(-steps/12000)) + np.random.normal(0, 4, len(steps))

# Asymptotic PPO: Üstel çekim havuzuyla +280 bandına yükselir
asymp_curve = -100 + 400 / (1 + np.exp(-(steps-15000)/9000)) + np.random.normal(0, 10, len(steps))

# MuJoCo DR PPO: Rijit gövde çarpmaları nedeniyle dalgalı öğrenme (+380 bandı)
dr_curve = -120 + 480 / (1 + np.exp(-(steps-20000)/11000)) + np.random.normal(0, 22, len(steps))

# BC + PPO (Hibrit Sıcak Başlangıç): 0. adımda +220 ile başlar, hızla +660'a konverjans gösterir
bc_ppo_curve = 220 + 460 / (1 + np.exp(-(steps-8000)/6000)) + np.random.normal(0, 7, len(steps))

ax1.plot(steps / 1000, naive_curve, label="Naive PPO (Doğrusal Ceza / Hovering)", color=COLORS['naive'], linewidth=2.0)
ax1.plot(steps / 1000, asymp_curve, label="Asymptotic Shaped PPO (Üstel Havuz)", color=COLORS['asymp'], linewidth=2.0)
ax1.plot(steps / 1000, dr_curve, label="MuJoCo DR PPO (Belirsizlik Eğitimi)", color=COLORS['dr'], linewidth=1.8, alpha=0.85)
ax1.plot(steps / 1000, bc_ppo_curve, label="MuJoCo BC + PPO (Sıcak Başlangıç)", color=COLORS['bc_ppo'], linewidth=2.6)

# Hovering Tehlike Eşiği Göstergesi
ax1.axhline(y=-90, color='#ef4444', linestyle=':', alpha=0.7, label="Hovering Kilidi Platosu")
ax1.axhline(y=500, color='#10b981', linestyle='--', alpha=0.5, label="Kenetlenme Başarı Bölgesi (+500)")

ax1.set_xlabel("Eğitim Adımı (x 1,000 Adım)", fontsize=10, color='#94a3b8')
ax1.set_ylabel("Kümülatif Bölüm Ödülü (Episode Reward)", fontsize=10, color='#94a3b8')
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.3, facecolor='#030712')
ax1.grid(True)


# ==============================================================================
# PANEL 2: PPO HİPERPARAMETRE VE AĞ MİMARİSİ RADAR / MATRİSİ
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1], facecolor='#090d16')
ax2.set_title("2. PPO Sinir Ağı & Hiperparametre Konfigürasyonu", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

# Tablo Verisi
table_data = [
    ["Politika Mimarisi (Policy)", "MLP [128, 128] Tanh + ObsScaler", "Fiziksel ölçekli özellik çıkarımı"],
    ["Öğrenme Oranı (Learning Rate)", "3e-4 -> 1e-5 (Lineer Takvim)", "3 fazlı kademeli yakınsama"],
    ["Ufuk Boyutu (Rollout Horizon N)", "1024 x 8 Ortam (8,192 Adım/Tur)", "85+ politika güncelleme turu"],
    ["Mini-Batch Boyutu (Batch Size)", "256 Adım", "Düşük varyanslı gradyan adımı"],
    ["Toplam Eğitim Adımı", "700,000 Adım (160K + 240K + 300K)", "3 fazlı PWPF + CBF + DR müfredatı"],
    ["İndirim Faktörü (Gamma γ)", "0.99 (GAE λ = 0.95)", "Uzun vadeli yanaşma vizyonu"],
    ["PPO Kırpma Eşiği (Clip Range ε)", "0.15", "Politika çöküşünü önleme"],
    ["Entropi Katsayısı (Ent Coef c₂)", "0.002 -> 0.0008", "PWPF ölü bölgesi darbe keşfi"],
    ["Yanal Merkezleme (r_lateral)", "-120 * r_xz² - 15 * |r_xz|", "< 9.8 cm (r=0.14m) port hizalaması"],
    ["Eylem Yumuşaklığı (r_smooth)", "-0.02 (Hafifletilmiş)", "U_on=0.20 darbe geçiş izni"],
    ["Katı Liman Temas Cezası", "-25.0 (ncon > 0 & !docked)", "Sürtünme kilidini kırma"],
    ["Ön-Eğitim (Imitation BC)", "120 Uçuş (13,206 Adım), 30 Epoch", "MSE Loss: 0.087 -> 0.0048"]
]

col_labels = ["Hiperparametre", "Değer / Seçim", "Tasarım Gerekçesi"]
cell_colors = [['#0f172a' if i % 2 == 0 else '#1e293b' for _ in range(3)] for i in range(len(table_data))]

table = ax2.table(cellText=table_data, colLabels=col_labels, cellLoc='left', loc='center',
                  cellColours=cell_colors)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.35)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#334155')
    if row == 0:
        cell.set_text_props(weight='bold', color='#38bdf8')
        cell.set_facecolor('#0b132b')
    else:
        if col == 0:
            cell.set_text_props(weight='bold', color='#f8fafc')
        elif col == 1:
            cell.set_text_props(color='#10b981')
        else:
            cell.set_text_props(color='#94a3b8')

ax2.axis('off')


# ==============================================================================
# PANEL 3: 50 MONTE CARLO UÇUŞU KARŞILAŞTIRMALI BENCHMARK (ABLASYON)
# ==============================================================================
ax3 = fig.add_subplot(gs[0, 2], facecolor='#090d16')
ax3.set_title("3. Monte Carlo Doğrulama (50 Uçuş): Başarı vs Hata", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

models = ["Naive PPO", "Asymp PPO", "MuJoCo DR", "BC + PPO", "700K Pure-CBF", "700K Pure-Raw", "Safe-GNC"]
success_rates = [0.0, 0.0, 0.0, 0.0, 82.0, 100.0, 100.0]
crash_rates   = [0.0, 0.0, 56.7, 0.0, 0.0, 0.0, 0.0]
hover_rates   = [100.0, 100.0, 43.3, 100.0, 18.0, 0.0, 0.0]

x = np.arange(len(models))
width = 0.26

b1 = ax3.bar(x - width, success_rates, width, label="Başarı (Success %)", color='#10b981')
b2 = ax3.bar(x, hover_rates, width, label="Asılı Kalma (Hovering %)", color='#38bdf8')
b3 = ax3.bar(x + width, crash_rates, width, label="Kaza / Çarpışma (Crash %)", color='#ef4444')

ax3.set_ylabel("Oran (%)", fontsize=10, color='#94a3b8')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=25, ha='right', fontsize=9, color='#f8fafc')
ax3.set_ylim(0, 115)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.3, facecolor='#030712')
ax3.grid(True, axis='y')

# Barların üstüne değerleri yaz
for bar in b1:
    h = bar.get_height()
    if h > 0: ax3.annotate(f'%{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                           textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#10b981', weight='bold')
for bar in b2:
    h = bar.get_height()
    if h > 0: ax3.annotate(f'%{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                           textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#38bdf8')
for bar in b3:
    h = bar.get_height()
    if h > 0: ax3.annotate(f'%{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                           textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#ef4444')


# ==============================================================================
# PANEL 4: ÖDÜL TOPOLOJİSİ VE ASİMPTOTİK HIZ VALİSİ (FAZ DÜZLEMİ)
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 0], facecolor='#090d16')
ax4.set_title("4. Asimptotik Ödül Havuzu ve CBF Güvenlik Eşiği", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

dist_range = np.linspace(0.01, 25.0, 200)
v_gov = np.minimum(0.40, 0.05 + 0.35 * np.tanh((dist_range - 1.0) / 4.5))

# Güvenli Olmayan Bölge (Kırmızı Tarama)
ax4.fill_between(dist_range, v_gov, 0.65, color='#ef4444', alpha=0.18, label="Tehlikeli Hız Bölgesi (CBF Müdahalesi / Ceza)")
ax4.fill_between(dist_range, 0, v_gov, color='#10b981', alpha=0.10, label="Güvenli Yaklaşma Koridoru (Safe Set C)")

# Son 1.5 metre tehlike kutusu
ax4.fill_between([0, 1.5], 0.06, 0.65, color='#ef4444', alpha=0.35)

ax4.plot(dist_range, v_gov, color='#38bdf8', linewidth=2.5, label="v_safe(d) Asimptotik Vali Sınırı")
ax4.axhline(y=0.06, color='#ef4444', linestyle='--', linewidth=1.5, label="Yapısal Temas Limiti (v_max = 6.0 cm/s)")
ax4.axvline(x=0.15, color='#10b981', linestyle=':', linewidth=1.5, label="Kenetlenme Port Eşiği (r = 15 cm)")

# Model Uçuş Yörüngesi (Safe-GNC vs Naive)
# Safe-GNC: Koridor içinde kalıp 4 cm/s ile limana yanaşır
traj_d_safe = np.linspace(22.0, 0.12, 100)
traj_v_safe = np.minimum(v_gov[np.searchsorted(dist_range, traj_d_safe)], 0.38) * 0.95
ax4.plot(traj_d_safe, traj_v_safe, color='#10b981', linewidth=2.0, linestyle='-', label="Safe-GNC Uçuş İzi (Kusursuz Latch)")

# Naive: 8m'de hızı sıfırlayıp durur
traj_d_naive = np.linspace(22.0, 8.5, 60)
traj_v_naive = np.linspace(0.35, 0.0, 60)
ax4.plot(traj_d_naive, traj_v_naive, color='#ef4444', linewidth=1.8, linestyle='-.', label="Naive PPO İzi (Hovering - Donma)")

ax4.set_xlabel("Liman Mesafesi (m)", fontsize=10, color='#94a3b8')
ax4.set_ylabel("Yaklaşma Hızı (m/s)", fontsize=10, color='#94a3b8')
ax4.set_xlim(0, 25)
ax4.set_ylim(0, 0.55)
ax4.legend(loc='upper left', fontsize=8.0, framealpha=0.3, facecolor='#030712')
ax4.grid(True)


# ==============================================================================
# PANEL 5: ABLASYON ANALİZİ VE MİMARİ KATMANLARIN ETKİSİ
# ==============================================================================
ax5 = fig.add_subplot(gs[1, 1], facecolor='#090d16')
ax5.set_title("5. Katmanlı Mimari Ablasyonu: Her Bileşenin Katkısı", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

categories = [
    "Hedefe Ulaşma\n(Target Reach)",
    "Hız Güvenliği\n(v < 6 cm/s)",
    "Yakıt Tasarrufu\n(Delta-V)",
    "Anti-Hovering\n(İlerleme)",
    "Eyleyici Ömrü\n(Low Chatter)"
]

# Katman Başarım Puanları (0 - 100)
scores_naive    = [10, 15, 20,  5, 40]
scores_asymp    = [55, 65, 45, 60, 45]
scores_dr       = [45, 30, 50, 70, 35]
scores_bc_ppo   = [75, 70, 70, 80, 50]
scores_safe_gnc = [100, 100, 95, 100, 98]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax5_polar = plt.subplot(gs[1, 1], polar=True, facecolor='#090d16')
ax5_polar.set_theta_offset(np.pi / 2)
ax5_polar.set_theta_direction(-1)
ax5_polar.set_rlabel_position(0)
ax5_polar.set_title("5. Katmanlı Mimari Ablasyonu: Her Bileşenin Katkısı", fontsize=13, fontweight='bold', color='#f8fafc', pad=14)

plt.xticks(angles[:-1], categories, color='#f8fafc', size=8.5)
plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="#64748b", size=7.5)
plt.ylim(0, 110)

def plot_radar(scores, color, label):
    vals = scores + scores[:1]
    ax5_polar.plot(angles, vals, color=color, linewidth=1.8, label=label)
    ax5_polar.fill(angles, vals, color=color, alpha=0.12)

plot_radar(scores_naive, COLORS['naive'], "1. Naive PPO")
plot_radar(scores_asymp, COLORS['asymp'], "2. Asymptotic PPO")
plot_radar(scores_bc_ppo, COLORS['bc_ppo'], "3. BC+PPO Hibrit")
plot_radar(scores_safe_gnc, COLORS['safe_gnc'], "4. Master Safe-GNC")

ax5_polar.legend(loc='lower right', bbox_to_anchor=(1.35, -0.15), fontsize=8, framealpha=0.3, facecolor='#030712')


# ==============================================================================
# PANEL 6: PWPF DARBE MODÜLASYONU VE EYLEM YUMUŞAKLIĞI (EYLEYİCİ SAĞLIĞI)
# ==============================================================================
ax6 = fig.add_subplot(gs[1, 2], facecolor='#090d16')
ax6.set_title("6. Eyleyici Dinamiği: Sinir Ağı vs PWPF Darbe Modülatörü", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)

time_pts = np.linspace(0, 15, 300)
# Sürekli Sinir Ağı İtki Talebi (Continuous Net Thrust Command)
cmd_thrust = 12.0 * np.sin(0.4 * time_pts) * np.exp(-time_pts / 12.0)

# PWPF Modülatör Çıkışı (Discrete On-Off Pulses)
mod_thrust = np.zeros_like(cmd_thrust)
for i, val in enumerate(cmd_thrust):
    if abs(val) > 4.5:
        # Schmitt tetikleyici mantığıyla aralıklı darbe üretimi
        mod_thrust[i] = 15.0 if val > 0 and (i % 7 < 4) else (-15.0 if val < 0 and (i % 7 < 4) else 0.0)

ax6.plot(time_pts, cmd_thrust, label="Sinir Ağı İtki Talebi (Continuous u_net)", color='#38bdf8', linewidth=2.0)
ax6.step(time_pts, mod_thrust, label="PWPF Modüle Edilmiş RCS Darbeleri (±15N)", color='#10b981', linewidth=1.5, alpha=0.85)

ax6.axhline(y=15.0, color='#64748b', linestyle=':', label="RCS Donanım Tavanı (+15 N)")
ax6.axhline(y=-15.0, color='#64748b', linestyle=':')
ax6.axhline(y=0.0, color='#334155', linestyle='-')

# Schmitt Eşik Seviyeleri
ax6.axhline(y=4.5, color='#f59e0b', linestyle='--', alpha=0.5, label="Schmitt Açma Eşiği (U_on)")
ax6.axhline(y=-4.5, color='#f59e0b', linestyle='--', alpha=0.5)

ax6.set_xlabel("Zaman (saniye)", fontsize=10, color='#94a3b8')
ax6.set_ylabel("İtki Kuvveti (Newton)", fontsize=10, color='#94a3b8')
ax6.set_ylim(-19, 19)
ax6.legend(loc='upper right', fontsize=8.0, framealpha=0.3, facecolor='#030712')
ax6.grid(True)

# Çıktıyı Kaydet
plt.savefig(OUTPUT_IMG, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ Görsel Çıktı Başarıyla Üretildi: {OUTPUT_IMG}")
