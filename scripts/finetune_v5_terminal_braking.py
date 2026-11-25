#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Champion-PPO v5.1 — Terminal Frenleme & Saf Politika (Pure-Raw + S3 Gürültü) İnce Ayarı
----------------------------------------------------------------------------------------------------
v5 (480K) zaten 360° Fly-Around (S5) dahil tüm 6 senaryoda 14 cm limana ulaşıyordu (dist=0.125m),
ancak r_speed cezası 6.5 cm/s hızında sadece -0.002 verdiği için Pure-Raw modunda 0.065 m/s ile
(limitten 5 mm/s fazla) temas ediyordu.

Bu Aşama-2 ince ayarı:
  1. Son 3 metrede v_ref = 0.036 m/s + güçlü doğrusal+kuadratik hız fren cezası ekler.
  2. Eğitim epizotlarının %75'inde CBF'yi kapatır (50% pure_raw, 25% pure_pwpf, 25% cbf_pwpf) ->
     sinir ağı frenlemeyi bizzat öğrenir, CBF bağımlılığı çöker.
  3. Sensör gürültüsü DR zarfını S3 (6 cm / 1.8 cm/s + ±5 cm bias + gecikme) seviyesine genişletir.
  4. Held-Out OOD Early Stopping ile en yüksek OOD (Pure-Raw + CBF+PWPF) checkpoint'ini kaydeder.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_config import CONFIG
from train_mujoco_champion_v5 import (
    AntiOverfitOrbitalEnv,
    OODEarlyStoppingCallback,
    compute_flyaround_waypoint_and_phi,
    build_noisy_obs,
    evaluate_heldout_ood_fast,
    linear_schedule,
    V5_BEST_OOD_PATH,
    V5_REPORT_PATH,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
V51_STAGE1_BACKUP = os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_stage1_480k.zip")


class AntiOverfitOrbitalEnvV51(AntiOverfitOrbitalEnv):
    """Güçlü terminal fren valisi + %75 kalkansız eğitim + genişletilmiş S3 gürültü DR'si."""

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        # %50 Pure-Raw, %25 Pure-PWPF, %25 CBF+PWPF -> sinir ağı frenlemeyi kendi öğrenir!
        m_r = float(self.rng.random())
        if m_r < 0.50:
            self.ep_mode = "pure_raw"
        elif m_r < 0.75:
            self.ep_mode = "pure_pwpf"
        else:
            self.ep_mode = "cbf_pwpf"

        # Gürültü & bias DR zarfını S3 (5cm/1.5cm/s + ±4cm bias) üstüne genişlet
        if self.ep_use_dr:
            self.pos_std = float(self.rng.uniform(0.005, 0.060))
            self.vel_std = float(self.rng.uniform(0.002, 0.018))
            self.pos_bias = self.rng.uniform(-0.045, 0.045, size=3)
            self.delay_steps = 1 if (self.rng.random() < 0.45) else 0
            self.action_queue = [np.zeros(3, dtype=np.float32)] * self.delay_steps
        return obs, info

    def step(self, action):
        if self.env.current_step == self.shock_step:
            self.env.data.qvel[:3] += self.shock_vec
            mujoco.mj_forward(self.env.model, self.env.data)

        cmd_act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.delay_steps > 0:
            self.action_queue.append(cmd_act)
            exec_act = self.action_queue.pop(0)
        else:
            exec_act = cmd_act

        obs_for_ctrl = self.last_obs if self.last_obs is not None else self.env._get_obs()
        cmd_force = exec_act * CONFIG.max_thrust

        if self.ep_mode == "cbf_pwpf":
            safe_force, _, _ = self.cbf.filter_action(obs_for_ctrl[:3], obs_for_ctrl[3:6], cmd_force, self.env.n)
            mod_force, _ = self.pwpf.modulate(safe_force / CONFIG.max_thrust, distance=float(obs_for_ctrl[7]))
            final_force = mod_force * self.axis_fault
        elif self.ep_mode == "pure_pwpf":
            mod_force, _ = self.pwpf.modulate(cmd_force / CONFIG.max_thrust, distance=float(obs_for_ctrl[7]))
            final_force = mod_force * self.axis_fault
        else:
            final_force = cmd_force * self.axis_fault

        applied_act = np.clip(final_force / CONFIG.max_thrust, -1.0, 1.0)
        clean_obs, _, terminated, truncated, info = self.env.step(applied_act)

        rel_pos = clean_obs[:3].astype(np.float64)
        vel = clean_obs[3:6].astype(np.float64)
        dist = float(clean_obs[7])
        speed = float(clean_obs[8])
        lat_err = float(np.hypot(rel_pos[0], rel_pos[2]))

        wp, phi, is_flyaround = compute_flyaround_waypoint_and_phi(rel_pos)
        progress = self.prev_phi - phi
        self.prev_phi = phi

        r_prog = np.clip(0.55 * progress, -0.8, 0.8)

        to_wp = wp - rel_pos
        d_wp = float(np.linalg.norm(to_wp)) + 1e-6
        u_dir = to_wp / d_wp
        v_close = float(np.dot(vel, u_dir))

        if is_flyaround:
            v_ref = min(0.48, 0.14 + 0.34 * np.tanh(d_wp / 5.0))
            excess = max(0.0, speed - v_ref)
            r_speed = -min(0.6, 3.0 * excess ** 2)
        else:
            # Terminal güvenli hedef hız: 3.6 cm/s (6.0 cm/s limitinin 2.4 cm/s altında güvenlik marjı!)
            if dist < 1.5:
                v_ref = 0.036
            else:
                v_ref = 0.036 + 0.41 * np.tanh((dist - 1.5) / 5.0)
            excess = max(0.0, speed - v_ref)
            if dist < 3.0:
                # Güçlü doğrusal + kuadratik frenleme valisi (0.065 m/s'de -0.72/adım net uyarı!)
                r_speed = -min(1.2, 22.0 * excess + 100.0 * (excess ** 2))
            else:
                r_speed = -min(0.7, 4.0 * excess + 12.0 * (excess ** 2))

        r_terminal = 0.0
        if not is_flyaround:
            if dist < 3.5:
                r_terminal -= min(0.25, 0.22 * (lat_err ** 2) + 0.06 * lat_err)
            if dist < 1.5 and rel_pos[1] > 0.0:
                if lat_err < 0.12 and 0.020 <= v_close <= 0.048 and speed <= 0.050:
                    r_terminal += 0.16
                elif v_close < 0.012:
                    r_terminal -= 0.12

        r_contact = -0.85 if (self.env.data.ncon > 0 and not info.get("docked_success", False)) else 0.0
        r_smooth = -0.040 * float(np.mean((cmd_act - self.prev_action) ** 2)) - 0.010 * float(np.mean(cmd_act ** 2))
        self.prev_action = cmd_act.copy()
        r_time = -0.010

        reward = float(r_prog + r_speed + r_terminal + r_contact + r_smooth + r_time)

        if info.get("docked_success", False):
            # Yumuşak temas (v <= 0.045 m/s) için ekstra temiz iniş bonusu
            soft_bonus = 3.0 if speed <= 0.048 else 0.0
            reward += 16.0 + soft_bonus + 2.0 * max(0.0, 1.0 - self.env.current_step / 450.0)
        elif info.get("crashed", False):
            reward -= 14.0

        if self.pos_std > 0.0 or self.vel_std > 0.0:
            obs_out = build_noisy_obs(clean_obs, self.env.n, self.rng, self.pos_std, self.vel_std, self.pos_bias)
        else:
            obs_out = clean_obs.copy()

        self.last_obs = obs_out
        return obs_out, reward, terminated, truncated, info


def make_env_v51(rank, base_seed=55000):
    def _init():
        env = AntiOverfitOrbitalEnvV51(max_steps=450, dr_prob=0.88)
        env.reset(seed=base_seed + rank * 149)
        return env
    return _init


def main(extra_steps=480_000, n_envs=8):
    import shutil
    t0 = time.time()
    print("=" * 95)
    print("⚡ ORBITAL-RL // CHAMPION-PPO v5.1 (PURE-RAW FRENLEME + S3 GÜRÜLTÜ İNCE AYARI)")
    print("=" * 95)

    if not os.path.exists(V51_STAGE1_BACKUP):
        shutil.copy2(V5_BEST_OOD_PATH, V51_STAGE1_BACKUP)

    try:
        vec_env = SubprocVecEnv([make_env_v51(i) for i in range(n_envs)], start_method="forkserver")
    except Exception:
        vec_env = DummyVecEnv([make_env_v51(i) for i in range(n_envs)])

    model = PPO.load(V5_BEST_OOD_PATH, env=vec_env, device="cpu")
    model.ent_coef = 0.0025
    model.lr_schedule = linear_schedule(1.2e-4, 1.5e-5)

    stage1_snap = {
        k: p.detach().cpu().clone()
        for k, p in model.policy.named_parameters()
        if ("policy_net" in k or "action_net" in k)
    }

    init_ood = evaluate_heldout_ood_fast(model)
    print(
        f"🔍 [Stage-1 Başlangıç] Held-Out OOD Başarı: %{init_ood['ood_success_rate']:.1f} | "
        f"Kaza: %{init_ood['ood_crash_rate']:.1f} | Δv: {init_ood['ood_mean_dv']:.2f} | "
        f"Skor: {init_ood['composite_score']}"
    )

    cb = OODEarlyStoppingCallback(eval_freq=48_000, patience=4, bc_snapshot=stage1_snap)
    cb.best_score = init_ood["composite_score"]
    cb.last_eval_step = int(model.num_timesteps)

    print(f"🔥 Aşama-2 Eğitimi Başlıyor (+{extra_steps:,} adım, %75 kalkansız, güçlü terminal fren valisi)...")
    model.learn(total_timesteps=extra_steps, reset_num_timesteps=False, callback=cb)
    vec_env.close()

    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 95)
    print(f"🏁 v5.1 Tamamlandı ({elapsed}s) | En İyi OOD Adımı: {cb.best_step:,} | En İyi Skor: {cb.best_score}")
    print("=" * 95)

    # Raporu güncelle
    report = {}
    if os.path.exists(V5_REPORT_PATH):
        with open(V5_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
    report["stage2_v51"] = {
        "elapsed_seconds": elapsed,
        "total_timesteps": int(model.num_timesteps),
        "best_ood_step": int(cb.best_step),
        "best_ood_composite": cb.best_score,
        "eval_history": cb.history,
    }
    with open(V5_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
