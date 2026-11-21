"""
Orbital-RL: 30 Dakikalık Paralel Monte Carlo Stres Testi
-------------------------------------------------------
Mevcut Champion-PPO modelini (ppo_mujoco_champion.zip) hiç eğitmeden,
8 paralel çekirdek üzerinde 30 dakika boyunca rastgele senaryolarda test eder.
Çift ID'li uçuşlar: Nominal MuJoCo (CBF + PWPF)
Tek ID'li uçuşlar : Gürültülü + Domain Randomization (Kütle ±%10, İtki ±%15, Sensör Gürültüsü)
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
from orbital_mujoco_env import OrbitalMujocoEnv
from pwpf_modulator import PWPFModulator3Axis
from cbf_qp_shield import MujocoCBFQPShield
from orbital_config import CONFIG

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion.zip")
OUT_JSON = os.path.join(DIR_PATH, "monte_carlo_30min_results.json")

_WORKER_MODEL = None
_WORKER_ENV = None
_WORKER_PWPF = None
_WORKER_CBF = None


def _init_worker():
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_PWPF, _WORKER_CBF
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_MODEL = PPO.load(MODEL_PATH, device="cpu")
    _WORKER_ENV = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    _WORKER_PWPF = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
    _WORKER_CBF = MujocoCBFQPShield(safety_margin_pct=0.10)


def _run_batch(args):
    start_ep, count, seed_offset = args
    env = _WORKER_ENV
    model = _WORKER_MODEL
    pwpf = _WORKER_PWPF
    cbf = _WORKER_CBF

    stats = {
        "nom": {"total": 0, "succ": 0, "crash": 0, "hover": 0, "dist_sum": 0.0, "vel_sum": 0.0},
        "dr": {"total": 0, "succ": 0, "crash": 0, "hover": 0, "dist_sum": 0.0, "vel_sum": 0.0},
    }

    for i in range(count):
        ep_id = start_ep + i
        seed = seed_offset + ep_id
        noisy_dr = (ep_id % 2 == 1)
        bucket = stats["dr"] if noisy_dr else stats["nom"]

        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        pwpf.reset()

        if noisy_dr:
            env.m = CONFIG.chaser_mass * float(rng.uniform(0.90, 1.10))
            env.max_thrust = CONFIG.max_thrust * float(rng.uniform(0.85, 1.15))
        else:
            env.m = CONFIG.chaser_mass
            env.max_thrust = CONFIG.max_thrust

        done = trunc = False
        while not (done or trunc):
            in_obs = obs.copy()
            if noisy_dr:
                in_obs[:3] += rng.normal(0.0, 0.004, 3)
                in_obs[3:6] += rng.normal(0.0, 0.0015, 3)
                in_obs[7] = float(np.linalg.norm(in_obs[:3]))
                in_obs[8] = float(np.linalg.norm(in_obs[3:6]))

            act, _ = model.predict(in_obs, deterministic=True)
            force = act * CONFIG.max_thrust
            force, _, _ = cbf.filter_action(obs[:3], obs[3:6], force, env.n)
            force, _ = pwpf.modulate(force / CONFIG.max_thrust, distance=float(info["distance"]))
            obs, _, done, trunc, info = env.step(np.clip(force / CONFIG.max_thrust, -1.0, 1.0))

        bucket["total"] += 1
        bucket["dist_sum"] += float(info["distance"])
        bucket["vel_sum"] += float(info["speed"])
        if info.get("docked_success"):
            bucket["succ"] += 1
        elif info.get("crashed"):
            bucket["crash"] += 1
        else:
            bucket["hover"] += 1

    return stats


def wilson_ci(succ, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = succ / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0))


def main(duration_sec=1800, n_workers=8, batch_size=100):
    print("=" * 80)
    print(f"ORBITAL-RL // {duration_sec // 60} DAKİKALIK PARALEL MONTE CARLO TESTİ ({n_workers} Çekirdek)")
    print(f"Model: {MODEL_PATH}")
    print("=" * 80)

    t_start = time.time()
    last_log = t_start
    seed_offset = 100_000
    next_ep = 0

    agg = {
        "nom": {"total": 0, "succ": 0, "crash": 0, "hover": 0, "dist_sum": 0.0, "vel_sum": 0.0},
        "dr": {"total": 0, "succ": 0, "crash": 0, "hover": 0, "dist_sum": 0.0, "vel_sum": 0.0},
    }

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        while (time.time() - t_start) < duration_sec:
            tasks = []
            for _ in range(n_workers):
                tasks.append((next_ep, batch_size, seed_offset))
                next_ep += batch_size

            batch_results = pool.map(_run_batch, tasks)
            for res in batch_results:
                for key in ("nom", "dr"):
                    for field in agg[key]:
                        agg[key][field] += res[key][field]

            now = time.time()
            elapsed = now - t_start
            total_eps = agg["nom"]["total"] + agg["dr"]["total"]

            if (now - last_log) >= 60.0 or elapsed >= duration_sec:
                last_log = now
                nom_p = (agg["nom"]["succ"] / max(1, agg["nom"]["total"])) * 100.0
                dr_p = (agg["dr"]["succ"] / max(1, agg["dr"]["total"])) * 100.0
                print(
                    f"[{elapsed:6.1f}s] Toplam: {total_eps:,} uçuş | "
                    f"Nominal: %{nom_p:.2f} ({agg['nom']['succ']}/{agg['nom']['total']}) | "
                    f"DR+Gürültü: %{dr_p:.2f} ({agg['dr']['succ']}/{agg['dr']['total']})",
                    flush=True,
                )

    elapsed = time.time() - t_start
    total_eps = agg["nom"]["total"] + agg["dr"]["total"]
    total_succ = agg["nom"]["succ"] + agg["dr"]["succ"]
    total_crash = agg["nom"]["crash"] + agg["dr"]["crash"]
    total_hover = agg["nom"]["hover"] + agg["dr"]["hover"]

    nom_ci = wilson_ci(agg["nom"]["succ"], agg["nom"]["total"])
    dr_ci = wilson_ci(agg["dr"]["succ"], agg["dr"]["total"])
    all_ci = wilson_ci(total_succ, total_eps)

    summary = {
        "duration_seconds": round(elapsed, 2),
        "total_episodes": total_eps,
        "overall": {
            "success_rate": round((total_succ / max(1, total_eps)) * 100.0, 4),
            "crash_rate": round((total_crash / max(1, total_eps)) * 100.0, 4),
            "hover_rate": round((total_hover / max(1, total_eps)) * 100.0, 4),
            "ci_95": [round(all_ci[0], 4), round(all_ci[1], 4)],
            "success_count": total_succ,
            "crash_count": total_crash,
            "hover_count": total_hover,
        },
        "nominal_cbf_pwpf": {
            "episodes": agg["nom"]["total"],
            "success_rate": round((agg["nom"]["succ"] / max(1, agg["nom"]["total"])) * 100.0, 4),
            "crash_rate": round((agg["nom"]["crash"] / max(1, agg["nom"]["total"])) * 100.0, 4),
            "hover_rate": round((agg["nom"]["hover"] / max(1, agg["nom"]["total"])) * 100.0, 4),
            "ci_95": [round(nom_ci[0], 4), round(nom_ci[1], 4)],
            "mean_final_distance": round(agg["nom"]["dist_sum"] / max(1, agg["nom"]["total"]), 4),
            "mean_final_speed": round(agg["nom"]["vel_sum"] / max(1, agg["nom"]["total"]), 4),
        },
        "dr_noisy_cbf_pwpf": {
            "episodes": agg["dr"]["total"],
            "success_rate": round((agg["dr"]["succ"] / max(1, agg["dr"]["total"])) * 100.0, 4),
            "crash_rate": round((agg["dr"]["crash"] / max(1, agg["dr"]["total"])) * 100.0, 4),
            "hover_rate": round((agg["dr"]["hover"] / max(1, agg["dr"]["total"])) * 100.0, 4),
            "ci_95": [round(dr_ci[0], 4), round(dr_ci[1], 4)],
            "mean_final_distance": round(agg["dr"]["dist_sum"] / max(1, agg["dr"]["total"]), 4),
            "mean_final_speed": round(agg["dr"]["vel_sum"] / max(1, agg["dr"]["total"]), 4),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🏁 30 DAKİKALIK MONTE CARLO TAMAMLANDI ({total_eps:,} Uçuş / {elapsed:.1f}s)")
    print(f"   Genel Başarı   : %{summary['overall']['success_rate']:.2f} (%95 CI: {all_ci[0]:.2f}% - {all_ci[1]:.2f}%) | Kaza: %{summary['overall']['crash_rate']:.2f} | Hover: %{summary['overall']['hover_rate']:.2f}")
    print(f"   Nominal        : %{summary['nominal_cbf_pwpf']['success_rate']:.2f} (%95 CI: {nom_ci[0]:.2f}% - {nom_ci[1]:.2f}%)")
    print(f"   DR + Gürültü   : %{summary['dr_noisy_cbf_pwpf']['success_rate']:.2f} (%95 CI: {dr_ci[0]:.2f}% - {dr_ci[1]:.2f}%)")
    print(f"📁 Kaydedildi: {OUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
