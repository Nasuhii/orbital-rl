"""
Adım 1: MuJoCo Fiziğinin CW Analitik Çözümüyle Doğrulanması (Sıfır İtki / Balistik Uçuş)
--------------------------------------------------------------------------------------
Bu script, kontrolsüz (u = 0) balistik yörüngeyi 100 saniye boyunca MuJoCo ve
Clohessy-Wiltshire (CW) kapalı form analitik çözümüyle yan yana çalıştırır.
Coriolis ve yerçekimi gradyanı kuvvetlerinin doğruluğunu sayısal hata metrikleriyle
(Maksimum Hata, RMSE, Son Konum/Hız Sapması, Açısal Hız) raporlar.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
import mujoco


def cw_analytical_solution(state0, n, t):
    """
    3D Clohessy-Wiltshire (Hill) Diferansiyel Denklemlerinin Kapalı Form Analitik Çözümü
    state0: [x0, y0, z0, vx0, vy0, vz0]
    n     : Ortalama hareket frekansı (rad/s)
    t     : Geçen süre (saniye)
    """
    x0, y0, z0, vx0, vy0, vz0 = state0
    s_nt = np.sin(n * t)
    c_nt = np.cos(n * t)

    # Radyal Eksen (x, vx)
    x = (4.0 - 3.0 * c_nt) * x0 + (s_nt / n) * vx0 + (2.0 * (1.0 - c_nt) / n) * vy0
    vx = 3.0 * n * s_nt * x0 + c_nt * vx0 + 2.0 * s_nt * vy0

    # Along-track Eksen (y, vy)
    y = 6.0 * (s_nt - n * t) * x0 + y0 - (2.0 * (1.0 - c_nt) / n) * vx0 + ((4.0 * s_nt - 3.0 * n * t) / n) * vy0
    vy = 6.0 * n * (c_nt - 1.0) * x0 - 2.0 * s_nt * vx0 + (4.0 * c_nt - 3.0) * vy0

    # Cross-track Düzlem Dışı Eksen (z, vz)
    z = z0 * c_nt + (s_nt / n) * vz0
    vz = -z0 * n * s_nt + vz0 * c_nt

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)


def run_physics_verification(t_total=100.0):
    env = OrbitalMujocoEnv()
    
    # Gerçekçi Balistik Başlangıç Durumu
    s0 = np.array([2.5, 20.0, -1.8, 0.03, -0.15, 0.02], dtype=np.float64)

    env.reset(seed=42)
    env.data.qpos[0] = s0[0]
    env.data.qpos[1] = s0[1]
    env.data.qpos[2] = s0[2]
    env.data.qpos[3] = 1.0
    env.data.qpos[4:7] = 0.0
    env.data.qvel[0] = s0[3]
    env.data.qvel[1] = s0[4]
    env.data.qvel[2] = s0[5]
    env.data.qvel[3:6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    dt_step = env.frame_skip * env.model.opt.timestep # 5 * 0.1s = 0.5s
    num_steps = int(t_total / dt_step)

    t_hist = [0.0]
    mj_hist = [np.concatenate([env.data.qpos[:3].copy(), env.data.qvel[:3].copy()])]
    cw_hist = [s0.copy()]
    ang_vel_hist = [env.data.qvel[3:6].copy()]

    zero_action = np.zeros(3)

    for k in range(num_steps):
        env.step(zero_action)
        t_now = (k + 1) * dt_step
        t_hist.append(t_now)
        s_mj = np.concatenate([env.data.qpos[:3].copy(), env.data.qvel[:3].copy()])
        s_cw = cw_analytical_solution(s0, env.n, t_now)
        mj_hist.append(s_mj)
        cw_hist.append(s_cw)
        ang_vel_hist.append(env.data.qvel[3:6].copy())

    mj_arr = np.array(mj_hist)
    cw_arr = np.array(cw_hist)
    ang_vel_arr = np.array(ang_vel_hist)

    pos_err = np.linalg.norm(mj_arr[:, :3] - cw_arr[:, :3], axis=1)
    vel_err = np.linalg.norm(mj_arr[:, 3:6] - cw_arr[:, 3:6], axis=1)

    print("\n" + "=" * 70)
    print("🛰️  ADIM 1: SIFIR KONTROLLÜ BALİSTİK YÖRÜNGE DOĞRULAMASI")
    print(f"Fizik Motoru : Google DeepMind MuJoCo v3.17 vs CW Analitik Çözüm")
    print(f"Süre         : {t_total:.1f} saniye ({num_steps} Gymnasium adımı / 1000 alt-adım)")
    print(f"Başlangıç s0 : pos={s0[:3]}, vel={s0[3:6]}")
    print("=" * 70)
    print(f"Maksimum Konum Hatası (Max Error) : {np.max(pos_err):.6e} m ({np.max(pos_err)*1000:.3f} mm)")
    print(f"Ortalama Konum Hatası (RMSE)      : {np.sqrt(np.mean(pos_err**2)):.6e} m ({np.sqrt(np.mean(pos_err**2))*1000:.3f} mm)")
    print(f"Nihai Konum Hatası (t=100s)       : {pos_err[-1]:.6e} m ({pos_err[-1]*1000:.3f} mm)")
    print(f"Maksimum Hız Hatası (Max Error)   : {np.max(vel_err):.6e} m/s")
    print(f"Nihai Hız Hatası (t=100s)         : {vel_err[-1]:.6e} m/s")
    print(f"Maksimum Açısal Hız (Parazit Tork): {np.max(np.linalg.norm(ang_vel_arr, axis=1)):.6e} rad/s")
    print(f"Nihai Quaternion                  : {env.data.qpos[3:7]}")
    print(f"MuJoCo Bitiş Konumu               : {mj_arr[-1, :3]}")
    print(f"CW Analitik Bitiş Konumu          : {cw_arr[-1, :3]}")
    print("=" * 70 + "\n")

    return {
        "max_pos_err": float(np.max(pos_err)),
        "rmse_pos_err": float(np.sqrt(np.mean(pos_err**2))),
        "final_pos_err": float(pos_err[-1]),
        "max_vel_err": float(np.max(vel_err)),
        "final_vel_err": float(vel_err[-1]),
        "max_ang_vel": float(np.max(np.linalg.norm(ang_vel_arr, axis=1)))
    }


if __name__ == "__main__":
    run_physics_verification(100.0)
