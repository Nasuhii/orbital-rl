"""
Orbital-RL: Genelleştirilmiş (Anti-Overfitting) 1M Adım PPO Eğitimi & OOD Doğrulaması
-------------------------------------------------------------------------------------
Dağılıma aşırı uyumu (distribution overfitting) kırmak için:
1. Geniş Başlangıç Zarfı: 0.6m - 45.0m mesafe, ±6m yanal sapma, ters yönlü ilk hızlar.
2. Ağır Domain Randomization: Kütle ±%25, İtki ±%20, eksen asimetrisi, sensör gürültüsü.
3. Uçuş Ortası Şok Bozuntusu: Rastgele hız darbesi enjeksiyonu.
4. Önce/Sonra OOD (Zarf Dışı) Stres Testi + Standart 50 Uçuşluk Benchmark.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_config import CONFIG
from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from train_mujoco_champion_ppo import (
    linear_schedule,
    evaluate_pure_mujoco,
    CHAMPION_MODEL_PATH,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
GEN_MODEL_PATH = os.path.join(DIR_PATH, "models", "ppo_mujoco_champion_generalized.zip")
OOD_REPORT_PATH = os.path.join(DIR_PATH, "generalization_1m_report.json")


class WideGeneralizationEnv(gym.Wrapper):
    def __init__(self, max_steps=450, dr_prob=0.85, shock_prob=0.20):
        env = OrbitalMujocoEnv(max_steps=max_steps)
        super().__init__(env)
        self.dr_prob = dr_prob
        self.shock_prob = shock_prob
        self.nominal_mass = CONFIG.chaser_mass
        self.nominal_thrust = CONFIG.max_thrust
        self.cbf = MujocoCBFQPShield(safety_margin_pct=0.10)
        self.pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
        self.axis_scale = np.ones(3, dtype=np.float64)
        self.ep_use_dr = True
        self.ep_use_cbf = True
        self.ep_shock_step = -1
        self.prev_dist = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)

    def _apply_sensor_noise(self, obs):
        if not self.ep_use_dr:
            return obs
        noisy = obs.copy()
        noisy[:3] += np.random.normal(0.0, 0.005, size=3)
        noisy[3:6] += np.random.normal(0.0, 0.002, size=3)
        noisy[7] = float(np.linalg.norm(noisy[:3]))
        noisy[8] = float(np.linalg.norm(noisy[3:6]))
        return noisy.astype(np.float32)

    def reset(self, seed=None, **kwargs):
        _, info = self.env.reset(seed=seed, **kwargs)
        self.pwpf.reset()

        self.ep_use_dr = bool(np.random.rand() < self.dr_prob)
        self.ep_use_cbf = bool(np.random.rand() < 0.95)
        self.ep_shock_step = int(np.random.randint(15, 120)) if np.random.rand() < self.shock_prob else -1

        if self.ep_use_dr:
            self.env.m = self.nominal_mass * float(np.random.uniform(0.75, 1.25))
            self.env.max_thrust = self.nominal_thrust * float(np.random.uniform(0.80, 1.20))
            self.axis_scale = np.random.uniform(0.85, 1.00, size=3)
        else:
            self.env.m = self.nominal_mass
            self.env.max_thrust = self.nominal_thrust
            self.axis_scale = np.ones(3, dtype=np.float64)

        # Genişletilmiş Çok-Rejimli Müfredat (%25 yakın, %40 orta/standart, %35 ekstrem OOD)
        r = np.random.rand()
        if r < 0.25:
            y_init = np.random.uniform(0.6, 4.0)
            x_init = np.random.uniform(-0.5, 0.5)
            z_init = np.random.uniform(-0.5, 0.5)
            vy_init = np.random.uniform(-0.08, 0.04)
            vx_init = np.random.uniform(-0.03, 0.03)
            vz_init = np.random.uniform(-0.03, 0.03)
        elif r < 0.65:
            y_init = np.random.uniform(4.0, 26.0)
            x_init = np.random.uniform(-3.5, 3.5)
            z_init = np.random.uniform(-2.8, 2.8)
            vy_init = np.random.uniform(-0.24, 0.08)
            vx_init = np.random.uniform(-0.06, 0.06)
            vz_init = np.random.uniform(-0.06, 0.06)
        else:
            y_init = np.random.uniform(26.0, 45.0)
            x_init = np.random.uniform(-6.0, 6.0)
            z_init = np.random.uniform(-4.5, 4.5)
            vy_init = np.random.uniform(-0.25, 0.12)
            vx_init = np.random.uniform(-0.08, 0.08)
            vz_init = np.random.uniform(-0.08, 0.08)

        self.env.data.qpos[0] = x_init
        self.env.data.qpos[1] = y_init
        self.env.data.qpos[2] = z_init
        self.env.data.qvel[0] = vx_init
        self.env.data.qvel[1] = vy_init
        self.env.data.qvel[2] = vz_init

        import mujoco
        mujoco.mj_forward(self.env.model, self.env.data)

        obs = self.env._get_obs()
        self.prev_dist = float(obs[7])
        self.prev_action = np.zeros(3, dtype=np.float32)
        info["distance"] = self.prev_dist
        info["speed"] = float(obs[8])
        return self._apply_sensor_noise(obs), info

    def step(self, action):
        if self.env.current_step == self.ep_shock_step:
            self.env.data.qvel[0] += np.random.uniform(-0.03, 0.03)
            self.env.data.qvel[1] += np.random.uniform(-0.02, 0.03)
            self.env.data.qvel[2] += np.random.uniform(-0.03, 0.03)

        clean_obs = self.env._get_obs()
        cmd_action = np.clip(action, -1.0, 1.0)
        force = cmd_action * self.nominal_thrust

        if self.ep_use_cbf:
            force, _, _ = self.cbf.filter_action(
                clean_obs[:3], clean_obs[3:6], force, self.env.n
            )

        force, _ = self.pwpf.modulate(
            force / self.nominal_thrust, distance=float(clean_obs[7])
        )
        force = force * self.axis_scale

        applied_action = np.clip(force / self.nominal_thrust, -1.0, 1.0)
        obs, _, terminated, truncated, info = self.env.step(applied_action)

        dist = float(obs[7])
        speed = float(obs[8])
        vy = float(obs[4])

        progress = self.prev_dist - dist
        self.prev_dist = dist
        prog_gain = 140.0 if dist < 2.0 else 95.0
        r_progress = prog_gain * progress

        lat_err_sq = float(obs[0] ** 2 + obs[2] ** 2)
        lat_err = float(np.sqrt(lat_err_sq))
        if dist < 0.6:
            r_lateral = -180.0 * lat_err_sq - 25.0 * lat_err
            if lat_err < 0.05:
                r_lateral += 6.0
        elif dist < 3.0:
            r_lateral = -120.0 * lat_err_sq - 15.0 * lat_err
        else:
            r_lateral = -30.0 * lat_err_sq

        v_target = 0.045 if dist < 1.8 else min(0.45, 0.05 + 0.40 * np.tanh((dist - 1.0) / 5.0))
        r_speed = -35.0 * max(0.0, speed - v_target) ** 2

        r_creep = 0.0
        if dist < 1.8:
            if lat_err < 0.075 and -0.052 <= vy <= -0.028:
                r_creep += 28.0
            elif vy > -0.022:
                r_creep -= 40.0
            if dist < 0.45 and speed < 0.015:
                r_creep -= 55.0
            if abs(obs[1]) < 0.45 and lat_err > 0.075 and vy < -0.020:
                r_creep -= 40.0

        r_time = -0.15
        r_smooth = -0.02 * float(np.sum((cmd_action - self.prev_action) ** 2))
        self.prev_action = cmd_action.copy()

        total_reward = r_progress + r_lateral + r_speed + r_creep + r_time + r_smooth

        if not info.get("docked_success", False) and self.env.data.ncon > 0:
            total_reward -= 25.0

        if info.get("docked_success", False):
            time_bonus = 100.0 * max(0.0, 1.0 - (info["step"] / 450.0))
            total_reward += 800.0 + time_bonus
        elif info.get("crashed", False):
            total_reward -= 350.0

        return self._apply_sensor_noise(obs), float(total_reward), terminated, truncated, info


def make_wide_env(rank, base_seed=30000):
    def _init():
        env = WideGeneralizationEnv(max_steps=450, dr_prob=0.85, shock_prob=0.20)
        env.reset(seed=base_seed + rank * 101)
        return env
    return _init


def evaluate_ood(model, num_episodes=100, seed_base=88000):
    """Zarf Dışı (Out-of-Distribution) Zorlu Genelleme Testi: 28-44m, ±5.5m sapma, ters ilk hız, ±%25 DR."""
    env = OrbitalMujocoEnv(max_steps=450)
    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.10)

    succ = crash = hover = 0
    for ep in range(1, num_episodes + 1):
        rng = np.random.default_rng(seed_base + ep * 37)
        _, info = env.reset(seed=seed_base + ep * 37)
        pwpf.reset()

        env.m = CONFIG.chaser_mass * float(rng.uniform(0.75, 1.25))
        env.max_thrust = CONFIG.max_thrust * float(rng.uniform(0.80, 1.20))

        env.data.qpos[0] = float(rng.uniform(-5.5, 5.5))
        env.data.qpos[1] = float(rng.uniform(28.0, 44.0))
        env.data.qpos[2] = float(rng.uniform(-4.0, 4.0))
        env.data.qvel[0] = float(rng.uniform(-0.07, 0.07))
        env.data.qvel[1] = float(rng.uniform(-0.22, 0.10))
        env.data.qvel[2] = float(rng.uniform(-0.07, 0.07))

        import mujoco
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        info["distance"] = float(obs[7])

        done = trunc = False
        while not (done or trunc):
            in_obs = obs.copy()
            in_obs[:3] += rng.normal(0.0, 0.005, 3)
            in_obs[3:6] += rng.normal(0.0, 0.002, 3)
            in_obs[7] = float(np.linalg.norm(in_obs[:3]))
            in_obs[8] = float(np.linalg.norm(in_obs[3:6]))

            act, _ = model.predict(in_obs, deterministic=True)
            force = act * CONFIG.max_thrust
            force, _, _ = cbf.filter_action(obs[:3], obs[3:6], force, env.n)
            force, _ = pwpf.modulate(force / CONFIG.max_thrust, distance=float(info["distance"]))
            obs, _, done, trunc, info = env.step(np.clip(force / CONFIG.max_thrust, -1.0, 1.0))

        if info.get("docked_success"):
            succ += 1
        elif info.get("crashed"):
            crash += 1
        else:
            hover += 1

    return {
        "episodes": num_episodes,
        "success_rate": round(succ / num_episodes * 100.0, 2),
        "crash_rate": round(crash / num_episodes * 100.0, 2),
        "hover_rate": round(hover / num_episodes * 100.0, 2),
        "success_count": succ,
        "crash_count": crash,
        "hover_count": hover,
    }


def main(total_steps=1_000_000, n_envs=8):
    print("=" * 80)
    print("ORBITAL-RL // 1M ADIM GENELLEŞTİRME (ANTI-OVERFITTING) EĞİTİMİ")
    print("=" * 80)

    base_model = PPO.load(CHAMPION_MODEL_PATH, device="cpu")
    print("\n🔍 [1/4] Eğitim ÖNCESİ OOD (Zarf Dışı 28-44m + DR) Testi (100 uçuş)...")
    ood_before = evaluate_ood(base_model, num_episodes=100)
    print(f"   -> ÖNCE OOD Başarı: %{ood_before['success_rate']:.1f} ({ood_before['success_count']}/100) | Kaza: %{ood_before['crash_rate']:.1f} | Hover: %{ood_before['hover_rate']:.1f}")

    try:
        vec_env = SubprocVecEnv([make_wide_env(i) for i in range(n_envs)], start_method="forkserver")
    except Exception:
        vec_env = DummyVecEnv([make_wide_env(i) for i in range(n_envs)])

    model = PPO.load(CHAMPION_MODEL_PATH, env=vec_env, device="cpu")
    model.ent_coef = 0.001
    model.lr_schedule = linear_schedule(5e-5, 1e-5)

    print(f"\n🔥 [2/4] 1,000,000 Adım Geniş-Zarf + Ağır DR Eğitimi Başlıyor...")
    t0 = time.time()
    model.learn(total_timesteps=total_steps, reset_num_timesteps=False)
    elapsed = time.time() - t0
    print(f"   -> Eğitim Tamamlandı ({elapsed:.1f}s)")
    vec_env.close()

    model.save(GEN_MODEL_PATH)
    print(f"💾 Genelleştirilmiş Model Kaydedildi: {GEN_MODEL_PATH}")

    print("\n🎯 [3/4] Standart 50 Uçuşluk Saf MuJoCo Testi...")
    res_raw = evaluate_pure_mujoco(model, mode="pure_raw", num_eval_episodes=50)
    res_pwpf = evaluate_pure_mujoco(model, mode="pure_pwpf", num_eval_episodes=50)
    res_cbf_pwpf = evaluate_pure_mujoco(model, mode="pure_cbf_pwpf", num_eval_episodes=50)

    print("\n🚀 [4/4] Eğitim SONRASI OOD (Zarf Dışı 28-44m + DR) Testi (100 uçuş)...")
    ood_after = evaluate_ood(model, num_episodes=100)
    print(f"   -> SONRA OOD Başarı: %{ood_after['success_rate']:.1f} ({ood_after['success_count']}/100) | Kaza: %{ood_after['crash_rate']:.1f} | Hover: %{ood_after['hover_rate']:.1f}")

    promoted = False
    if res_cbf_pwpf["success_rate"] >= 96.0 and ood_after["success_rate"] >= ood_before["success_rate"]:
        shutil.copy2(GEN_MODEL_PATH, CHAMPION_MODEL_PATH)
        promoted = True
        print(f"\n🏆 YENİ GENELLEŞTİRİLMİŞ ŞAMPİYON! {CHAMPION_MODEL_PATH} güncellendi.")

    report = {
        "training_steps": total_steps,
        "training_seconds": round(elapsed, 2),
        "promoted_to_champion": promoted,
        "ood_before": ood_before,
        "ood_after": ood_after,
        "standard_after": {
            "pure_raw": res_raw["success_rate"],
            "pure_pwpf": res_pwpf["success_rate"],
            "pure_cbf_pwpf": res_cbf_pwpf["success_rate"],
        },
    }
    with open(OOD_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📁 Rapor Kaydedildi: {OOD_REPORT_PATH}")


if __name__ == "__main__":
    main()
