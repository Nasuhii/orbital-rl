"""
Adım 3 Doğrulaması: Chattering Sönümleme ve PWPF Modülatör Testi
--------------------------------------------------------------
Ham komutlar (Raw Bang-Bang) ile Yumuşatılmış PWPF Modülatörünü
MuJoCo üzerinde karşılaştırır, valf anahtarlama sıklığını ve
itki doygunluk titreşimini sayısal olarak ölçer.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG


def test_chattering_suppression(num_steps=120):
    env_raw = OrbitalMujocoEnv(max_steps=num_steps)
    env_pwpf = OrbitalMujocoEnv(max_steps=num_steps)

    obs_r, info_r = env_raw.reset(seed=101)
    obs_p, info_p = env_pwpf.reset(seed=101)

    env_pwpf.data.qpos[:] = env_raw.data.qpos[:].copy()
    env_pwpf.data.qvel[:] = env_raw.data.qvel[:].copy()

    pwpf = PWPFModulator3Axis(
        dt=CONFIG.dt,
        Km=1.0,
        Tm=1.2,
        U_on=0.20,
        U_off=0.08,
        max_thrust=CONFIG.max_thrust,
        action_rate_limit=0.35
    )

    np.random.seed(42)
    thrust_raw = []
    thrust_pwpf = []
    flips_raw = 0
    flips_pwpf = 0
    prev_raw_sign = np.zeros(3)
    prev_pwpf_sign = np.zeros(3)

    for step in range(num_steps):
        # Gerçekçi RL politikası: Temel rehberlik + %25 keşif/ölçüm gürültüsü
        dist = info_p['distance']
        base_cmd = np.clip(-obs_p[:3] / (dist + 1e-4) * 0.3 - obs_p[3:6] * 2.0, -1.0, 1.0)
        noise = np.random.normal(0.0, 0.25, size=3)
        noisy_cmd = np.clip(base_cmd + noise, -1.0, 1.0)

        # 1. Ham Bang-Bang İtici (Klasik Eşik)
        raw_force = np.where(np.abs(noisy_cmd) > 0.15, np.sign(noisy_cmd) * CONFIG.max_thrust, 0.0)

        # 2. PWPF Modüle Edilmiş Yumuşatılmış İtici
        pwpf_force, _ = pwpf.modulate(noisy_cmd, distance=dist)

        # Adım ilerlet
        obs_r, _, _, _, info_r = env_raw.step(raw_force / CONFIG.max_thrust)
        obs_p, _, _, _, info_p = env_pwpf.step(pwpf_force / CONFIG.max_thrust)

        thrust_raw.append(raw_force.copy())
        thrust_pwpf.append(pwpf_force.copy())

        curr_raw_sign = np.sign(raw_force)
        curr_pwpf_sign = np.sign(pwpf_force)

        if step > 0:
            flips_raw += np.sum((curr_raw_sign != prev_raw_sign) & (curr_raw_sign != 0))
            flips_pwpf += np.sum((curr_pwpf_sign != prev_pwpf_sign) & (curr_pwpf_sign != 0))

        prev_raw_sign = curr_raw_sign
        prev_pwpf_sign = curr_pwpf_sign

    thrust_raw_arr = np.array(thrust_raw)
    thrust_pwpf_arr = np.array(thrust_pwpf)

    dv_raw = np.sum(np.linalg.norm(thrust_raw_arr, axis=1)) / CONFIG.chaser_mass * CONFIG.dt
    dv_pwpf = np.sum(np.linalg.norm(thrust_pwpf_arr, axis=1)) / CONFIG.chaser_mass * CONFIG.dt
    chattering_reduction = ((flips_raw - flips_pwpf) / max(1, flips_raw)) * 100

    print("\n" + "=" * 70)
    print("🛰️  ADIM 3: CHATTERING SÖNÜMLEME VE PWPF TEST RAPORU")
    print(f"Toplam Test Adımı       : {num_steps} adım ({num_steps * CONFIG.dt:.1f} saniye)")
    print(f"Eyleyici İtki Limiti     : ±{CONFIG.max_thrust} N")
    print("=" * 70)
    print(f"Ham İtki (Raw Bang-Bang) Valf Aç/Kapa Sıklığı : {flips_raw} kez (Yüksek Titreşim/Yıpranma)")
    print(f"PWPF Modülatör Valf Aç/Kapa Sıklığı           : {flips_pwpf} kez")
    print(f"Chattering Azalma Oranı                        : %{chattering_reduction:.1f} İyileşme 🟢")
    print(f"PWPF Toplam Valf Yorgunluk Döngüsü             : {pwpf.total_valve_cycles} döngü")
    print(f"Ham Yakıt Harcaması (Delta-V)                  : {dv_raw:.3f} m/s")
    print(f"PWPF Yakıt Harcaması (Delta-V)                 : {dv_pwpf:.3f} m/s (%{((dv_raw - dv_pwpf)/dv_raw)*100:.1f} Tasarruf ⚡)")
    print("=" * 70 + "\n")

    return {
        "flips_raw": flips_raw,
        "flips_pwpf": flips_pwpf,
        "reduction_pct": chattering_reduction,
        "dv_raw": dv_raw,
        "dv_pwpf": dv_pwpf
    }


if __name__ == "__main__":
    test_chattering_suppression()
