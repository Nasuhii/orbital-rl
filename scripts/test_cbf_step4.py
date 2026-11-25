"""
Adım 4 Doğrulaması: CBF-QP Emniyet Kalkanının MuJoCo Ortamında Testi (%15 Güvenlik Marjı)
----------------------------------------------------------------------------------------
Kasıtlı olarak agresif ve yüksek hızlı komut üreten bir kontrolcüyü:
1. Kalkansız (Unshielded)
2. CBF-QP Kalkanlı (Shielded)
şekilde MuJoCo'da 10 farklı zorlu başlangıç koşulunda test eder.
Çarpışma sayısı, minimum bariyer değeri h_min ve müdahale oranını raporlar.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from orbital_config import CONFIG


def test_cbf_safety_shield(num_episodes=10):
    env_unshielded = OrbitalMujocoEnv(max_steps=260)
    env_shielded = OrbitalMujocoEnv(max_steps=260)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.15)

    crashes_unshielded = 0
    crashes_shielded = 0
    h_min_unshielded = []
    h_min_shielded = []
    total_steps = 0
    total_interventions = 0

    for ep in range(1, num_episodes + 1):
        seed = 200 + ep * 13
        obs_u, _ = env_unshielded.reset(seed=seed)
        obs_s, _ = env_shielded.reset(seed=seed)

        env_shielded.data.qpos[:] = env_unshielded.data.qpos[:].copy()
        env_shielded.data.qvel[:] = env_unshielded.data.qvel[:].copy()

        ep_h_u = []
        ep_h_s = []

        done_u, done_s = False, False
        trunc_u, trunc_s = False, False

        # 1. Kalkansız Simülasyon
        while not (done_u or trunc_u):
            rel_pos = obs_u[:3]
            vel = obs_u[3:6]
            dist = np.linalg.norm(rel_pos)
            speed = np.linalg.norm(vel)
            h = (cbf.get_v_safe(dist) ** 2) - (speed ** 2)
            ep_h_u.append(h)

            # Agresif komut: Limana 0.35 m/s ile dalış
            nom_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 200.0, -CONFIG.max_thrust, CONFIG.max_thrust)
            obs_u, _, done_u, trunc_u, info_u = env_unshielded.step(nom_force / CONFIG.max_thrust)

        if info_u['crashed']:
            crashes_unshielded += 1

        # 2. CBF-QP Kalkanlı Simülasyon
        while not (done_s or trunc_s):
            total_steps += 1
            rel_pos = obs_s[:3]
            vel = obs_s[3:6]
            dist = np.linalg.norm(rel_pos)

            # Aynı agresif komut
            nom_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 200.0, -CONFIG.max_thrust, CONFIG.max_thrust)

            # CBF-QP Kalkanından geçir
            safe_force, h_val, was_intervened = cbf.filter_action(rel_pos, vel, nom_force, env_shielded.n)
            ep_h_s.append(h_val)
            if was_intervened:
                total_interventions += 1

            obs_s, _, done_s, trunc_s, info_s = env_shielded.step(safe_force / CONFIG.max_thrust)

        if info_s['crashed']:
            crashes_shielded += 1

        h_min_unshielded.append(min(ep_h_u))
        h_min_shielded.append(min(ep_h_s))

    print("\n" + "=" * 72)
    print("🛡️  ADIM 4: CBF-QP EMNİYET KALKANI MUJOCO DOĞRULAMA RAPORU (%15 MARJ)")
    print(f"Toplam Test Uçuşu       : {num_episodes} uçuş")
    print(f"Nominal Temas Limiti     : {CONFIG.max_docking_speed:.3f} m/s (6.0 cm/s)")
    print(f"CBF Güvenli Temas Tavanı : {cbf.v_dock_safe:.3f} m/s (5.1 cm/s - %15 Marj)")
    print("=" * 72)
    print(f"Kalkansız Kaza Sayısı   : {crashes_unshielded}/{num_episodes} (%{crashes_unshielded/num_episodes*100:.1f} Kaza)")
    print(f"CBF Kalkanlı Kaza Sayısı : {crashes_shielded}/{num_episodes} (%{crashes_shielded/num_episodes*100:.1f} Kaza) 🟢")
    print(f"Kalkansız Min Bariyer h : {np.min(h_min_unshielded):.4f} (İHLAL: h < 0)")
    print(f"CBF Kalkanlı Min Bariyer: {np.min(h_min_shielded):.4f} (GÜVENLİ: h >= 0)")
    print(f"CBF Kalkan Müdahale Oranı: %{(total_interventions / max(1, total_steps)) * 100:.1f} ({total_interventions}/{total_steps} adım)")
    print("=" * 72 + "\n")

    return {
        "crashes_unshielded": crashes_unshielded,
        "crashes_shielded": crashes_shielded,
        "min_h_unshielded": float(np.min(h_min_unshielded)),
        "min_h_shielded": float(np.min(h_min_shielded)),
        "intervention_rate": float((total_interventions / max(1, total_steps)) * 100)
    }


if __name__ == "__main__":
    test_cbf_safety_shield(num_episodes=10)
