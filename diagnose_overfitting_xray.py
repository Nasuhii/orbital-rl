#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Gerçek (Sızıntısız & Bug-Free) Overfitting Röntgen & OOD Diagnostik Aracı
------------------------------------------------------------------------------------
Mevcut PPO modellerini (v3 [%82], v4 [%100], generalized [2.4M]) hem In-Distribution
hem de 5 gerçek Out-of-Distribution (OOD) stres senaryosunda, hem CBF KAPALI (Pure-Raw)
hem de CBF+PWPF AÇIK modda ölçer.

Düzeltilen Test Bug'ları:
  1. Kütle DR'si gerçek MuJoCo model.body_mass + body_inertia + mj_setConst ile uygulanır.
  2. Sensör gürültüsü eklendikten sonra obs[6] (coriolis), obs[7] (dist), obs[8] (speed)
     gürültülü veriden yeniden hesaplanır (sıfır veri sızıntısı).
  3. İtici güç kaybı / eksen arızası PWPF SONRASINDA fiziksel kuvvete uygulanır.
  4. Hız şoku doğrudan data.qvel[:3] üzerine enjekte edilir.
  5. Critic Explained Variance (R²), Action Std (exp log_std), CVaR-10%, CBF müdahale
     oranı ve OBS_SCALES doygunluk (körlük) yüzdesi hesaplanır.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_config import CONFIG
from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
OUT_JSON = os.path.join(DIR_PATH, "overfitting_xray_report.json")


def set_real_mujoco_mass(env, base_mass_arr, base_inertia_arr, mass_factor):
    """MuJoCo rijit gövde kütlesi ve ataletini GERÇEKTEN günceller."""
    env.m = CONFIG.chaser_mass * mass_factor
    env.model.body_mass[env.chaser_body_id] = base_mass_arr[env.chaser_body_id] * mass_factor
    env.model.body_inertia[env.chaser_body_id] = base_inertia_arr[env.chaser_body_id] * mass_factor
    mujoco.mj_setConst(env.model, env.data)


def build_noisy_obs(clean_obs, n_val, rng, pos_std, vel_std, pos_bias):
    """Sızıntısız sensör gürültüsü: obs[6:9] gürültülü veriden yeniden hesaplanır."""
    noisy = clean_obs.copy()
    noisy[0:3] += pos_bias + rng.normal(0.0, pos_std, size=3)
    noisy[3:6] += rng.normal(0.0, vel_std, size=3)
    noisy[6] = -2.0 * n_val * noisy[3]
    noisy[7] = float(np.linalg.norm(noisy[0:3]))
    noisy[8] = float(np.linalg.norm(noisy[3:6]))
    return noisy.astype(np.float32)


def compute_discounted_returns(rewards, gamma=0.99):
    returns = np.zeros(len(rewards), dtype=np.float64)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def inspect_model_internals(model, label):
    """Model ağırlıklarından entropi / log_std / timestep bilgilerini çeker."""
    log_std = None
    action_std = None
    if hasattr(model.policy, "log_std"):
        ls = model.policy.log_std.detach().cpu().numpy()
        log_std = [round(float(x), 4) for x in ls]
        action_std = [round(float(np.exp(x)), 4) for x in ls]

    obs_scales = None
    fe = getattr(model.policy, "features_extractor", None)
    if fe is not None and hasattr(fe, "scales"):
        obs_scales = [round(float(x), 4) for x in fe.scales.detach().cpu().numpy()]

    return {
        "label": label,
        "num_timesteps": int(model.num_timesteps),
        "ent_coef": float(model.ent_coef) if isinstance(model.ent_coef, (int, float)) else str(model.ent_coef),
        "gamma": float(model.gamma),
        "log_std": log_std,
        "action_std": action_std,
        "mean_action_std": round(float(np.mean(action_std)), 4) if action_std else None,
        "obs_scales": obs_scales,
    }


def run_suite(model, scenario_id, mode="pure_raw", num_episodes=30, seed_base=50000):
    """
    scenario_id:
      - S1_nominal_id       : Nominal eğitim zarfı (y:18-25, x:±2.5, z:±1.8, vy:-0.22..-0.10)
      - S2_real_mass_fault  : Gerçek MuJoCo kütle ±%35 + post-PWPF itki verim kaybı (%60-100) + 1 eksen %40 arıza
      - S3_unleaked_noise   : 5cm konum / 1.5cm/s hız gürültüsü + ±4cm bias + 0.5s gecikme (CBF de gürültülü alır)
      - S4_hard_ood_escape  : 30-50m, x:±10m, z:±8m, KAÇIŞ hızı vy:[+0.15,+0.40], yanal ±0.20 m/s
      - S5_wide_angle_side  : Yandan / arkadan yaklaşma (y:[-12,+10], x:±12m, z:±6m — +Y eksen ezber testi)
      - S6_real_qvel_shock  : Nominal + t=25s (0.18 m/s) ve t=55s (0.14 m/s) GERÇEK qvel fizik şokları
    mode:
      - pure_raw            : Saf sinir ağı (CBF yok, PWPF yok)
      - cbf_pwpf            : CBF-QP + PWPF aktif
    """
    max_steps = 450 if scenario_id in ("S4_hard_ood_escape", "S5_wide_angle_side") else 300
    env = OrbitalMujocoEnv(max_steps=max_steps)
    base_mass = env.model.body_mass.copy()
    base_inertia = env.model.body_inertia.copy()

    cbf = MujocoCBFQPShield(safety_margin_pct=0.10)
    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)

    fe = getattr(model.policy, "features_extractor", None)
    obs_scales_np = fe.scales.detach().cpu().numpy() if (fe is not None and hasattr(fe, "scales")) else None

    succ = crash = hover = 0
    ep_returns = []
    final_dists = []
    final_speeds = []
    delta_vs = []
    smooth_mses = []
    cbf_step_ratios = []
    clamp_sat_ratios = []
    all_v_preds = []
    all_g_trues = []

    for ep in range(num_episodes):
        seed = seed_base + ep * 97
        rng = np.random.default_rng(seed)
        _, info = env.reset(seed=seed)
        pwpf.reset()

        # Varsayılan fizik
        mass_factor = 1.0
        axis_fault = np.ones(3, dtype=np.float64)
        pos_std = 0.0
        vel_std = 0.0
        pos_bias = np.zeros(3, dtype=np.float64)
        delay_steps = 0
        shocks = {}

        # 1) ÖNCE gerçek MuJoCo kütle & atalet güncellemesi (mj_setConst qpos'u sıfırlar, o yüzden önce çağrılır!)
        if scenario_id == "S2_real_mass_fault":
            mass_factor = float(rng.choice([rng.uniform(0.65, 0.80), rng.uniform(1.20, 1.35)]))
            base_scale = float(rng.uniform(0.65, 0.90))
            axis_fault = np.full(3, base_scale, dtype=np.float64)
            fault_axis = int(rng.integers(0, 3))
            axis_fault[fault_axis] *= 0.60  # Ek %40 tek eksen arızası

        # Nominal reset'in ürettiği qpos/qvel'i koru
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        set_real_mujoco_mass(env, base_mass, base_inertia, mass_factor)
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel

        if scenario_id == "S3_unleaked_noise":
            pos_std = 0.05
            vel_std = 0.015
            pos_bias = rng.uniform(-0.04, 0.04, size=3)
            delay_steps = 1

        elif scenario_id == "S4_hard_ood_escape":
            env.data.qpos[0] = float(rng.uniform(-10.0, 10.0))
            env.data.qpos[1] = float(rng.uniform(30.0, 50.0))
            env.data.qpos[2] = float(rng.uniform(-8.0, 8.0))
            env.data.qvel[0] = float(rng.uniform(-0.20, 0.20))
            env.data.qvel[1] = float(rng.uniform(0.15, 0.40))   # İstasyondan KAÇIŞ!
            env.data.qvel[2] = float(rng.uniform(-0.20, 0.20))

        elif scenario_id == "S5_wide_angle_side":
            # Yandan veya arkadan başlama (+Y tüpü dışında!)
            angle = float(rng.uniform(np.pi * 0.35, np.pi * 1.65))  # +Y'den en az 63 derece uzak!
            radius = float(rng.uniform(10.0, 22.0))
            env.data.qpos[0] = radius * np.sin(angle)
            env.data.qpos[1] = radius * np.cos(angle)
            env.data.qpos[2] = float(rng.uniform(-6.0, 6.0))
            env.data.qvel[0] = float(rng.uniform(-0.10, 0.10))
            env.data.qvel[1] = float(rng.uniform(-0.10, 0.10))
            env.data.qvel[2] = float(rng.uniform(-0.08, 0.08))

        elif scenario_id == "S6_real_qvel_shock":
            shocks = {
                50: rng.normal(0.0, 1.0, size=3),   # t = 25s
                110: rng.normal(0.0, 1.0, size=3),  # t = 55s
            }
            for k, vec in shocks.items():
                mag = 0.18 if k == 50 else 0.14
                shocks[k] = (vec / (np.linalg.norm(vec) + 1e-8)) * mag

        mujoco.mj_forward(env.model, env.data)
        clean_obs = env._get_obs()

        action_queue = [np.zeros(3, dtype=np.float32)] * delay_steps
        prev_act = np.zeros(3, dtype=np.float32)
        rewards = []
        v_preds = []
        act_diffs = []
        cbf_hits = 0
        sat_steps = 0
        total_dv = 0.0
        steps = 0
        done = trunc = False

        while not (done or trunc):
            # Gerçek fiziksel hız şoku
            if steps in shocks:
                env.data.qvel[:3] += shocks[steps]
                mujoco.mj_forward(env.model, env.data)
                clean_obs = env._get_obs()

            # Gözlem (S3'te sızıntısız gürültülü, diğerlerinde temiz)
            if pos_std > 0.0 or vel_std > 0.0:
                in_obs = build_noisy_obs(clean_obs, env.n, rng, pos_std, vel_std, pos_bias)
            else:
                in_obs = clean_obs.copy()

            # Clamp doygunluk kontrolü (OBS_SCALES körlüğü)
            if obs_scales_np is not None:
                if np.any(np.abs(in_obs / obs_scales_np) >= 4.99):
                    sat_steps += 1

            # Critic V(s) tahmini
            with torch.no_grad():
                obs_t = torch.as_tensor(in_obs[None, :], dtype=torch.float32)
                v_val = float(model.policy.predict_values(obs_t).cpu().numpy().ravel()[0])
            v_preds.append(v_val)

            # Politika kararı
            raw_act, _ = model.predict(in_obs, deterministic=True)
            act_diffs.append(float(np.mean((raw_act - prev_act) ** 2)))
            prev_act = raw_act.copy()

            if delay_steps > 0:
                action_queue.append(raw_act)
                exec_act = action_queue.pop(0)
            else:
                exec_act = raw_act

            cmd_force = np.clip(exec_act, -1.0, 1.0) * CONFIG.max_thrust

            if mode == "cbf_pwpf":
                # CBF de sızıntısız olarak aynı sensör gözlemini (in_obs) görür!
                safe_force, _, intervened = cbf.filter_action(in_obs[:3], in_obs[3:6], cmd_force, env.n)
                if intervened:
                    cbf_hits += 1
                mod_force, _ = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=float(in_obs[7]))
                final_force = mod_force * axis_fault
            else:
                final_force = cmd_force * axis_fault

            # Gerçek uygulanan kuvvet (post-PWPF fault dahil!)
            applied_action = np.clip(final_force / CONFIG.max_thrust, -1.0, 1.0)
            total_dv += (float(np.linalg.norm(applied_action * CONFIG.max_thrust)) / env.m) * CONFIG.dt

            clean_obs, rew, done, trunc, info = env.step(applied_action)
            rewards.append(float(rew))
            steps += 1

        if info.get("docked_success"):
            succ += 1
        elif info.get("crashed"):
            crash += 1
        else:
            hover += 1

        ep_returns.append(float(np.sum(rewards)))
        final_dists.append(float(info["distance"]))
        final_speeds.append(float(info["speed"]))
        delta_vs.append(total_dv)
        smooth_mses.append(float(np.mean(act_diffs)) if act_diffs else 0.0)
        cbf_step_ratios.append((cbf_hits / max(1, steps)) * 100.0)
        clamp_sat_ratios.append((sat_steps / max(1, steps)) * 100.0)

        g_trues = compute_discounted_returns(rewards, gamma=float(model.gamma))
        all_g_trues.extend(g_trues.tolist())
        all_v_preds.extend(v_preds)

    # Critic Explained Variance (R²)
    g_arr = np.array(all_g_trues, dtype=np.float64)
    v_arr = np.array(all_v_preds, dtype=np.float64)
    var_g = float(np.var(g_arr))
    ev = float(1.0 - np.var(g_arr - v_arr) / (var_g + 1e-8)) if var_g > 1e-8 else 0.0

    # CVaR-10% (en kötü %10 uçuş ortalaması)
    sorted_ret = np.sort(ep_returns)
    k_worst = max(1, int(np.ceil(0.10 * len(sorted_ret))))
    cvar10 = float(np.mean(sorted_ret[:k_worst]))

    return {
        "episodes": num_episodes,
        "success_rate": round(succ / num_episodes * 100.0, 1),
        "crash_rate": round(crash / num_episodes * 100.0, 1),
        "hover_rate": round(hover / num_episodes * 100.0, 1),
        "mean_return": round(float(np.mean(ep_returns)), 1),
        "cvar10_return": round(cvar10, 1),
        "critic_ev_r2": round(ev, 4),
        "mean_final_dist": round(float(np.mean(final_dists)), 3),
        "mean_final_speed": round(float(np.mean(final_speeds)), 4),
        "std_final_speed": round(float(np.std(final_speeds)), 6),
        "mean_dv": round(float(np.mean(delta_vs)), 3),
        "action_smooth_mse": round(float(np.mean(smooth_mses)), 5),
        "cbf_intervention_pct": round(float(np.mean(cbf_step_ratios)), 1),
        "obs_clamp_sat_pct": round(float(np.mean(clamp_sat_ratios)), 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Sadece belirli model anahtarını çalıştır (ör: v5_anti_overfit)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 95)
    print("🔬 ORBITAL-RL // GERÇEK (BUG-FREE & SIZINTISIZ) OVERFITTING RÖNTGENİ (ADIM 1)")
    print("=" * 95)

    model_specs = [
        ("v3_700k", os.path.join(MODELS_DIR, "ppo_mujoco_champion_v3_backup.zip")),
        ("v4_1.4m", os.path.join(MODELS_DIR, "ppo_mujoco_champion_v4.zip")),
        ("gen_2.4m", os.path.join(MODELS_DIR, "ppo_mujoco_champion_generalized.zip")),
        ("v5_anti_overfit", os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_best_ood.zip")),
    ]
    if args.only:
        model_specs = [m for m in model_specs if m[0] == args.only]

    scenarios = [
        ("S1_nominal_id",      "S1: Nominal ID (Eğitim Zarfı)"),
        ("S2_real_mass_fault", "S2: Gerçek Kütle ±%35 + İtki Arızası"),
        ("S3_unleaked_noise",  "S3: 5cm/1.5cm Gürültü + Bias + Gecikme"),
        ("S4_hard_ood_escape", "S4: 30-50m, ±10m, KAÇIŞ Hızı (vy>0)"),
        ("S5_wide_angle_side", "S5: Yandan/Arkadan Yaklaşma (+Y Dışı)"),
        ("S6_real_qvel_shock", "S6: Gerçek qvel Çift Hız Şoku"),
    ]

    full_report = {"models": {}}

    for m_key, m_path in model_specs:
        if not os.path.exists(m_path):
            print(f"[ATLA] Bulunamadı: {m_path}")
            continue

        model = PPO.load(m_path, device="cpu")
        internals = inspect_model_internals(model, m_key)
        print(f"\n📦 MODEL: {m_key} | Steps: {internals['num_timesteps']:,} | ent_coef: {internals['ent_coef']} | Action Std: {internals['action_std']}")
        print(f"   OBS_SCALES: {internals['obs_scales']}")
        print("-" * 108)
        print(f"{'Senaryo':40s} | {'Mod':8s} | {'Başarı':7s} | {'Kaza':6s} | {'Hover':6s} | {'CBF%':5s} | {'Sat%':5s} | {'EV(R²)':7s} | {'SpdStd':8s} | {'Δv':5s}")
        print("-" * 108)

        m_res = {"internals": internals, "results": {}}
        for s_id, s_label in scenarios:
            m_res["results"][s_id] = {}
            for mode in ("pure_raw", "cbf_pwpf"):
                r = run_suite(model, s_id, mode=mode, num_episodes=25)
                m_res["results"][s_id][mode] = r
                print(
                    f"{s_label:40s} | {mode:8s} | %{r['success_rate']:5.1f} | %{r['crash_rate']:4.1f} | "
                    f"%{r['hover_rate']:4.1f} | {r['cbf_intervention_pct']:4.1f}% | {r['obs_clamp_sat_pct']:4.1f}% | "
                    f"{r['critic_ev_r2']:7.3f} | {r['std_final_speed']:8.5f} | {r['mean_dv']:5.2f}",
                    flush=True,
                )

        # Özet Generalization Gap (S1 vs S2..S6 ortalaması)
        for mode in ("pure_raw", "cbf_pwpf"):
            id_sr = m_res["results"]["S1_nominal_id"][mode]["success_rate"]
            ood_srs = [m_res["results"][s_id][mode]["success_rate"] for s_id, _ in scenarios[1:]]
            ood_avg = round(float(np.mean(ood_srs)), 1)
            gap = round(id_sr - ood_avg, 1)
            m_res[f"summary_{mode}"] = {
                "id_success": id_sr,
                "ood_mean_success": ood_avg,
                "generalization_gap": gap,
            }
        print(
            f"   >>> ÖZET ({m_key}) | Pure-Raw: ID %{m_res['summary_pure_raw']['id_success']} vs OOD Ort %{m_res['summary_pure_raw']['ood_mean_success']} "
            f"(Gap: {m_res['summary_pure_raw']['generalization_gap']}%) | "
            f"CBF+PWPF: ID %{m_res['summary_cbf_pwpf']['id_success']} vs OOD Ort %{m_res['summary_cbf_pwpf']['ood_mean_success']} "
            f"(Gap: {m_res['summary_cbf_pwpf']['generalization_gap']}%)"
        )
        full_report["models"][m_key] = m_res

    if args.only and os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev.setdefault("models", {}).update(full_report["models"])
        full_report = prev
    full_report["elapsed_seconds"] = round(time.time() - t0, 2)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f"✅ Röntgen tamamlandı ({full_report['elapsed_seconds']}s) -> {OUT_JSON}")
    print("=" * 95)


if __name__ == "__main__":
    main()
