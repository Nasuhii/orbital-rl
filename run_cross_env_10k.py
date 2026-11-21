"""
Orbital-RL: 10,000 Uçuşluk Paralel Çapraz Motor (RK4 CW) Stres Testi
-------------------------------------------------------------------
MuJoCo'dan tamamen bağımsız OrbitalDockingEnv (RK4 Clohessy-Wiltshire)
ortamında 8 çekirdek ile 10,000 uçuşluk büyük çapraz doğrulama:
  1. 3,000 uçuş: Saf RK4 CW (Standart)
  2. 3,500 uçuş: 800 km İrtifa + 0.5s Gecikme + %70 İtki
  3. 3,500 uçuş: Ekstrem Zarf (30-50m, ±7m, Ters İlk Hız + ±%25 DR + Sensör Gürültüsü)
"""

import os
import sys
import time
import json
import math
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from orbital_env import OrbitalDockingEnv
from pwpf_modulator import PWPFModulator3Axis
from cbf_qp_shield import MujocoCBFQPShield
from orbital_config import CONFIG

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion.zip")
OUT_JSON = os.path.join(DIR_PATH, "cross_env_10k_results.json")

_MODEL = None


def _init_worker():
    global _MODEL
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _MODEL = PPO.load(MODEL_PATH, device="cpu")


def _to_obs9(state6, n_val):
    pos = state6[:3]
    vel = state6[3:6]
    return np.array([
        pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2],
        -2.0 * n_val * vel[0],
        float(np.linalg.norm(pos)),
        float(np.linalg.norm(vel)),
    ], dtype=np.float32)


def _run_chunk(args):
    mode, start_ep, count, seed_base = args
    altitude = 800_000.0 if mode == "delayed_800km" else 400_000.0
    max_steps = 450 if mode == "extreme_dr" else 300
    env = OrbitalDockingEnv(orbit_altitude=altitude, max_steps=max_steps)
    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.10)

    succ = crash = hover = 0
    for i in range(count):
        ep_id = start_ep + i
        seed = seed_base + ep_id
        rng = np.random.default_rng(seed)
        state, _ = env.reset(seed=seed)
        pwpf.reset()

        thrust_scale = 0.70 if mode == "delayed_800km" else 1.0
        delay_steps = 1 if mode == "delayed_800km" else 0
        env.m = CONFIG.chaser_mass

        if mode == "extreme_dr":
            env.m = CONFIG.chaser_mass * float(rng.uniform(0.75, 1.25))
            thrust_scale = float(rng.uniform(0.80, 1.20))
            env.state = np.array([
                rng.uniform(-7.0, 7.0),
                rng.uniform(30.0, 50.0),
                rng.uniform(-5.0, 5.0),
                rng.uniform(-0.08, 0.08),
                rng.uniform(-0.22, 0.12),
                rng.uniform(-0.08, 0.08),
            ], dtype=np.float64)
            state = env.state.copy()

        action_queue = [np.zeros(3, dtype=np.float32)] * delay_steps
        done = trunc = False
        while not (done or trunc):
            obs9 = _to_obs9(state, env.n)
            if mode == "extreme_dr":
                obs9[:3] += rng.normal(0.0, 0.005, 3)
                obs9[3:6] += rng.normal(0.0, 0.002, 3)
                obs9[7] = float(np.linalg.norm(obs9[:3]))
                obs9[8] = float(np.linalg.norm(obs9[3:6]))

            act, _ = _MODEL.predict(obs9, deterministic=True)
            if delay_steps > 0:
                action_queue.append(act)
                exec_act = action_queue.pop(0)
            else:
                exec_act = act

            force = exec_act * CONFIG.max_thrust * thrust_scale
            force, _, _ = cbf.filter_action(state[:3], state[3:6], force, env.n)
            force, _ = pwpf.modulate(force / CONFIG.max_thrust, distance=float(np.linalg.norm(state[:3])))
            state, _, done, trunc, _ = env.step(np.clip(force / CONFIG.max_thrust, -1.0, 1.0))

        dist = float(np.linalg.norm(state[:3]))
        spd = float(np.linalg.norm(state[3:6]))
        if dist <= CONFIG.docking_radius and spd <= CONFIG.max_docking_speed:
            succ += 1
        elif dist > CONFIG.max_radial_departure or (dist <= CONFIG.docking_radius and spd > CONFIG.max_docking_speed):
            crash += 1
        else:
            hover += 1

    return succ, crash, hover


def wilson_ci(succ, total, z=1.96):
    p = succ / max(1, total)
    denom = 1.0 + (z * z) / max(1, total)
    center = (p + (z * z) / (2.0 * max(1, total))) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * max(1, total))) / max(1, total))) / denom
    return (round(max(0.0, (center - half) * 100.0), 3), round(min(100.0, (center + half) * 100.0), 3))


def main(n_workers=8):
    print("=" * 80)
    print("ORBITAL-RL // 10,000 UÇUŞLUK ÇAPRAZ MOTOR (RK4 CW) MONTE CARLO TESTİ")
    print("=" * 80)
    t0 = time.time()

    scenarios = [
        ("rk4_standard", "1. Saf RK4 CW (Standart)", 3000, 200_000),
        ("delayed_800km", "2. 800km İrtifa + 0.5s Gecikme + %70 İtki", 3500, 300_000),
        ("extreme_dr", "3. Ekstrem (30-50m, ±7m, Ters Hız + ±%25 DR)", 3500, 400_000),
    ]

    summary = {"total_episodes": 10000, "scenarios": {}}
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for mode, label, total_eps, seed_base in scenarios:
            chunk = 125
            tasks = []
            for start in range(0, total_eps, chunk):
                cnt = min(chunk, total_eps - start)
                tasks.append((mode, start, cnt, seed_base))
            res_list = pool.map(_run_chunk, tasks)
            succ = sum(r[0] for r in res_list)
            crash = sum(r[1] for r in res_list)
            hover = sum(r[2] for r in res_list)
            ci = wilson_ci(succ, total_eps)
            sr = round(succ / total_eps * 100.0, 3)
            cr = round(crash / total_eps * 100.0, 3)
            hr = round(hover / total_eps * 100.0, 3)
            print(f"{label:48s} | {total_eps:,} uçuş | Başarı: %{sr:6.2f} ({succ}/{total_eps}) | %95 CI: {ci[0]}%-{ci[1]}% | Kaza: %{cr:.2f} | Hover: %{hr:.2f}", flush=True)
            summary["scenarios"][mode] = {
                "label": label,
                "episodes": total_eps,
                "success_count": succ,
                "crash_count": crash,
                "hover_count": hover,
                "success_rate": sr,
                "crash_rate": cr,
                "hover_rate": hr,
                "ci_95": list(ci),
            }

    elapsed = round(time.time() - t0, 2)
    summary["duration_seconds"] = elapsed
    tot_succ = sum(v["success_count"] for v in summary["scenarios"].values())
    tot_crash = sum(v["crash_count"] for v in summary["scenarios"].values())
    tot_hover = sum(v["hover_count"] for v in summary["scenarios"].values())
    summary["overall"] = {
        "success_count": tot_succ,
        "crash_count": tot_crash,
        "hover_count": tot_hover,
        "success_rate": round(tot_succ / 10000 * 100.0, 3),
        "crash_rate": round(tot_crash / 10000 * 100.0, 3),
        "hover_rate": round(tot_hover / 10000 * 100.0, 3),
        "ci_95": list(wilson_ci(tot_succ, 10000)),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(f"🏁 TOPLAM 10,000 UÇUŞ ({elapsed}s) | Genel Başarı: %{summary['overall']['success_rate']:.2f} ({tot_succ}/10,000) | Kaza: %{summary['overall']['crash_rate']:.2f} | Hover: %{summary['overall']['hover_rate']:.2f}")
    print(f"📁 Kaydedildi: {OUT_JSON}")


if __name__ == "__main__":
    main()
