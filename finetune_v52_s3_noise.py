#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Champion-PPO v5.2 — S3 (5cm/1.5cm/s Gürültü + ±4cm Bias + 0.5s Gecikme) Güçlendirme
------------------------------------------------------------------------------------------------
v5'in tek zayıf noktası olan S3 senaryosundaki:
  - %12 crash (pure_raw'da 0.060 m/s temas limitini aşma)
  - %24-%32 hover (0.15-0.25m'de ±4cm bias yüzünden asılı kalma)
sorunlarını çözmek için S3-odaklı + tüm OOD (S1..S6) korumalı ince ayar scripti.
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_config import CONFIG
from diagnose_overfitting_xray import run_suite
from train_mujoco_champion_v5 import (
    AntiOverfitOrbitalEnv,
    compute_flyaround_waypoint_and_phi,
    build_noisy_obs,
    linear_schedule,
    V5_BEST_OOD_PATH,
    V5_REPORT_PATH,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
V52_BACKUP_PATH = os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_pre_s3_backup.zip")


class S3HardenedOrbitalEnv(AntiOverfitOrbitalEnv):
    """S3 gürültü/bias/gecikme + terminal frenleme & anti-hover ince ayar ortamı."""

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        m_r = float(self.rng.random())
        if m_r < 0.55:
            self.ep_mode = "pure_raw"
        elif m_r < 0.75:
            self.ep_mode = "pure_pwpf"
        else:
            self.ep_mode = "cbf_pwpf"

        # %70 ihtimalle doğrudan ağır S3 gürültü + bias + gecikme profili uygula
        if self.rng.random() < 0.70:
            self.pos_std = float(self.rng.uniform(0.040, 0.060))
            self.vel_std = float(self.rng.uniform(0.012, 0.020))
            self.pos_bias = self.rng.uniform(-0.045, 0.045, size=3)
            self.delay_steps = 1
            self.action_queue = [np.zeros(3, dtype=np.float32)]
        elif self.ep_use_dr:
            self.pos_std = float(self.rng.uniform(0.005, 0.040))
            self.vel_std = float(self.rng.uniform(0.002, 0.014))
            self.pos_bias = self.rng.uniform(-0.030, 0.030, size=3)
            self.delay_steps = 1 if (self.rng.random() < 0.40) else 0
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
            # S3'te ±4cm bias & 5cm gürültü altında 0.060 m/s limitini aşmamak için 0.032 m/s hedefi
            v_ref = 0.032 if dist < 1.5 else (0.032 + 0.40 * np.tanh((dist - 1.5) / 5.0))
            excess = max(0.0, speed - v_ref)
            if dist < 3.0:
                r_speed = -min(1.4, 24.0 * excess + 110.0 * (excess ** 2))
            else:
                r_speed = -min(0.7, 4.0 * excess + 12.0 * (excess ** 2))

        r_terminal = 0.0
        if not is_flyaround:
            if dist < 3.5:
                r_terminal -= min(0.28, 0.25 * (lat_err ** 2) + 0.08 * lat_err)
            # 0.15m - 0.80m aralığında asılı kalmayı (hover) kır
            if 0.13 < dist < 0.90 and rel_pos[1] > 0.0:
                if v_close < 0.015:
                    r_terminal -= 0.20
                elif 0.018 <= v_close <= 0.044 and speed <= 0.048:
                    r_terminal += 0.18

        r_contact = -0.90 if (self.env.data.ncon > 0 and not info.get("docked_success", False)) else 0.0
        r_smooth = -0.18 * float(np.mean((cmd_act - self.prev_action) ** 2)) - 0.025 * float(np.mean(cmd_act ** 2))
        self.prev_action = cmd_act.copy()
        r_time = -0.012

        reward = float(r_prog + r_speed + r_terminal + r_contact + r_smooth + r_time)

        if info.get("docked_success", False):
            soft_bonus = 3.5 if speed <= 0.046 else 0.0
            reward += 17.0 + soft_bonus + 2.0 * max(0.0, 1.0 - self.env.current_step / 450.0)
        elif info.get("crashed", False):
            reward -= 16.0

        if self.pos_std > 0.0 or self.vel_std > 0.0:
            obs_out = build_noisy_obs(clean_obs, self.env.n, self.rng, self.pos_std, self.vel_std, self.pos_bias)
        else:
            obs_out = clean_obs.copy()

        self.last_obs = obs_out
        return obs_out, reward, terminated, truncated, info


def make_env_v52(rank, base_seed=66000):
    def _init():
        env = S3HardenedOrbitalEnv(max_steps=450, dr_prob=0.90)
        env.reset(seed=base_seed + rank * 157)
        return env
    return _init


def eval_s3_and_ood(model):
    """Gerçek S3 (20 epizot) + S2/S4/S5 (10'ar epizot) doğrusal kıyaslama."""
    s3_raw = run_suite(model, "S3_unleaked_noise", mode="pure_raw", num_episodes=20, seed_base=77000)
    s3_cbf = run_suite(model, "S3_unleaked_noise", mode="cbf_pwpf", num_episodes=20, seed_base=77000)
    s2 = run_suite(model, "S2_real_mass_fault", mode="pure_raw", num_episodes=10, seed_base=78000)
    s5 = run_suite(model, "S5_wide_angle_side", mode="pure_raw", num_episodes=10, seed_base=79000)
    score = (
        0.45 * s3_raw["success_rate"]
        + 0.25 * s3_cbf["success_rate"]
        + 0.15 * s2["success_rate"]
        + 0.15 * s5["success_rate"]
        - 1.2 * (s3_raw["crash_rate"] + s3_cbf["crash_rate"])
    )
    return {
        "s3_raw": s3_raw["success_rate"],
        "s3_raw_crash": s3_raw["crash_rate"],
        "s3_cbf": s3_cbf["success_rate"],
        "s2_raw": s2["success_rate"],
        "s5_raw": s5["success_rate"],
        "score": round(float(score), 2),
    }


def main(extra_steps=320_000, n_envs=8):
    from stable_baselines3.common.callbacks import BaseCallback

    t0 = time.time()
    print("=" * 95)
    print("📡 ORBITAL-RL // CHAMPION-PPO v5.2 (S3 20-EPİZOT DOĞRULAMALI GÜRÜLTÜ GÜÇLENDİRME)")
    print("=" * 95)

    src_model = V52_BACKUP_PATH if os.path.exists(V52_BACKUP_PATH) else V5_BEST_OOD_PATH
    try:
        vec_env = SubprocVecEnv([make_env_v52(i) for i in range(n_envs)], start_method="forkserver")
    except Exception:
        vec_env = DummyVecEnv([make_env_v52(i) for i in range(n_envs)])

    model = PPO.load(src_model, env=vec_env, device="cpu")
    model.ent_coef = 0.0018
    model.lr_schedule = linear_schedule(8.0e-5, 1.0e-5)

    init_res = eval_s3_and_ood(model)
    print(f"🔍 [Başlangıç] S3-Raw:%{init_res['s3_raw']:.1f} (Crash:%{init_res['s3_raw_crash']:.1f}) | S3-CBF:%{init_res['s3_cbf']:.1f} | S2:%{init_res['s2_raw']:.1f} | S5:%{init_res['s5_raw']:.1f} | Skor:{init_res['score']}")

    class S3Callback(BaseCallback):
        def __init__(self, init_score):
            super().__init__()
            self.best_score = init_score
            self.best_step = int(model.num_timesteps)
            self.last_eval = int(model.num_timesteps)
            self.no_imp = 0
            self.history = []

        def _on_step(self):
            if int(self.model.num_timesteps) - self.last_eval < 32_000:
                return True
            self.last_eval = int(self.model.num_timesteps)
            r = eval_s3_and_ood(self.model)
            self.history.append({"step": self.last_eval, **r})
            tag = ""
            if r["score"] > self.best_score:
                self.best_score = r["score"]
                self.best_step = self.last_eval
                self.no_imp = 0
                self.model.save(V5_BEST_OOD_PATH)
                tag = " ★ YENİ EN İYİ S3!"
            else:
                self.no_imp += 1
                tag = f" (gelişme yok {self.no_imp}/5)"
            print(f"📊 [@ {self.last_eval:,}] S3-Raw:%{r['s3_raw']:.1f} (Crash:%{r['s3_raw_crash']:.1f}) | S3-CBF:%{r['s3_cbf']:.1f} | S2:%{r['s2_raw']:.1f} | S5:%{r['s5_raw']:.1f} | Skor:{r['score']}{tag}")
            if self.no_imp >= 5:
                print("🛑 Early stopping: S3+OOD skoru 5 turdur artmadı.")
                return False
            return True

    cb = S3Callback(init_res["score"])
    model.learn(total_timesteps=extra_steps, reset_num_timesteps=False, callback=cb)
    vec_env.close()

    elapsed = round(time.time() - t0, 2)
    print(f"🏁 v5.2 Tamamlandı ({elapsed}s) | En İyi Adım: {cb.best_step:,} | En İyi Skor: {cb.best_score}")

    report = {}
    if os.path.exists(V5_REPORT_PATH):
        with open(V5_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
    report["stage3_v52_s3"] = {
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
