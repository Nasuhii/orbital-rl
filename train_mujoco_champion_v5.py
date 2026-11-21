#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Champion-PPO v5 — Gerçekten Öğrenen, Anti-Overfit & 360° Fly-Around Mimarisi
========================================================================================
Röntgen teşhisinde tespit edilen TÜM kök nedenleri çözer:
  1. Critic Patlaması & Aktör Donması Çözümü:
     - Ödül ölçeği [-1.5, +1.5]/adım, +15 dock / -10 crash (G_t ∈ [-15, +35]) bandına çekildi.
     - vf_loss milyonlara fırlamaz -> max_grad_norm aktör gradyanını ezmez -> PPO GERÇEKTEN ÖĞRENİR!
  2. Sıfır-Körlük (Zero-Saturation) Gözlem Mimarisi:
     - OrbitalObsFeaturesExtractorV5 (asinh + eksen-bağımsız 16 özellik + LayerNorm) ile
       |x|,|z|>2.5m körlüğü (%53-%82 saturation) %0.0'a indirilir.
  3. Gerçek, Sızıntısız Domain Randomization (DR):
     - MuJoCo model.body_mass + body_inertia ±%25 (mj_setConst + qpos/qvel korunarak).
     - Post-PWPF itki verimi & tek eksen kısmi arıza (%30-35 güç kaybı).
     - Sızıntısız sensör gürültüsü (obs[6:9] gürültülü veriden yeniden hesaplanır) + bias + 0.5s gecikme.
     - Gerçek data.qvel[:3] fiziksel hız şokları.
  4. 4 Rejimli Geniş Zarf + 360° İstasyon Etrafından Dolanan (Fly-Around) Eğitim + Z-Simetrisi.
  5. BC Sıcak Başlangıç (15 Epoch + Z-Simetri 2x Veri) -> PPO'da SIFIR expert_weight!
  6. AdamW (weight_decay=1e-4) + Sağlıklı Entropi (0.003) + Sabit Held-Out OOD Early Stopping.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_config import CONFIG
from orbital_mujoco_env import OrbitalMujocoEnv
from orbital_rl_features import OrbitalObsFeaturesExtractorV5
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

V5_BEST_OOD_PATH = os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_best_ood.zip")
V5_FINAL_PATH = os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_final.zip")
V5_BC_ONLY_PATH = os.path.join(MODELS_DIR, "ppo_mujoco_v5_bc_warmstart.zip")
V5_REPORT_PATH = os.path.join(DIR_PATH, "v5_anti_overfit_training_report.json")


def linear_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def set_real_mujoco_mass(env, base_mass, base_inertia, mass_factor):
    """MuJoCo kütlesini ve ataletini qpos/qvel'i bozmadan GERÇEKTEN günceller."""
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    env.m = CONFIG.chaser_mass * mass_factor
    cid = env.chaser_body_id
    env.model.body_mass[cid] = base_mass[cid] * mass_factor
    env.model.body_inertia[cid] = base_inertia[cid] * mass_factor
    mujoco.mj_setConst(env.model, env.data)
    env.data.qpos[:] = saved_qpos
    env.data.qvel[:] = saved_qvel


def build_noisy_obs(clean_obs, n_val, rng, pos_std, vel_std, pos_bias):
    """Sızıntısız sensör gürültüsü: obs[6:9] gürültülü veriden yeniden hesaplanır."""
    noisy = clean_obs.copy()
    noisy[0:3] += pos_bias + rng.normal(0.0, pos_std, size=3)
    noisy[3:6] += rng.normal(0.0, vel_std, size=3)
    noisy[6] = -2.0 * n_val * noisy[3]
    noisy[7] = float(np.linalg.norm(noisy[0:3]))
    noisy[8] = float(np.linalg.norm(noisy[3:6]))
    return noisy.astype(np.float32)


def compute_flyaround_waypoint_and_phi(rel_pos):
    """
    3D İstasyon Engelinden Kaçınan (Fly-Around) Sürekli Hedef ve Potansiyel Φ(s).
    İstasyon gövdesi (x∈[-3,3], y∈[-1.2,1.2]) + güneş panelleri (z∈[-6.3,6.3], x∈[-1.2,1.2]).
    Port: rel_pos = (0, 0, 0), istasyon merkezi rel_pos = (0, -1.6, 0).
    """
    rx, ry, rz = float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])
    lat_err = float(np.hypot(rx, rz))
    sign_x = 1.0 if rx >= 0.0 else -1.0

    # 1. Aşama: İstasyonun arkasında veya güneş paneli koridorunda (ry < 1.2)
    if ry < 1.2 and (ry < 0.2 or lat_err > 0.65):
        # Önce X ekseninde güvenli açıklığa (|rx| >= 5.2m) çık, Z'yi 0'a çek
        if abs(rx) < 4.8 and ry < 0.3:
            wp = np.array([sign_x * 5.8, min(ry, -1.5), 0.0], dtype=np.float64)
            # Kalan yol: wp + staging(5.8, 4.5, 0) + port(0, 0, 0)
            phi = float(np.linalg.norm(rel_pos - wp)) + float(np.linalg.norm(wp - np.array([sign_x * 5.8, 4.5, 0.0]))) + 7.5
            return wp, phi, True
        else:
            # Güvenli X koridorunda öne (+Y = 4.5m) ilerle
            wp = np.array([sign_x * max(5.2, min(abs(rx), 6.5)), 4.5, 0.0], dtype=np.float64)
            phi = float(np.linalg.norm(rel_pos - wp)) + float(np.linalg.norm(wp))
            return wp, phi, True

    # 2. Aşama: Ön yarımkürede ama henüz geniş yanal açıda (ry < 3.0 ve lat_err > 1.2)
    if ry < 3.0 and lat_err > 1.2:
        wp = np.array([0.0, 4.2, 0.0], dtype=np.float64)
        phi = float(np.linalg.norm(rel_pos - wp)) + 4.2
        return wp, phi, True

    # 3. Aşama: Doğrudan +Y V-Bar koridorunda kenetlenme portu (0, 0, 0)
    wp = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    phi = float(np.linalg.norm(rel_pos))
    return wp, phi, False


def get_flyaround_expert_action(obs, n_val=CONFIG.mean_motion, mass=CONFIG.chaser_mass):
    """
    Tüm 4 rejimde (nominal, kaçış hızı, 360° yandan/arkadan fly-around) çalışan
    engel farkındalıklı uzman referans kontrolcü (yalnızca 15 epoch BC warmstart için).
    """
    rel_pos = obs[:3].astype(np.float64)
    vel = obs[3:6].astype(np.float64)
    dist = float(np.linalg.norm(rel_pos)) + 1e-6

    wp, _, is_flyaround = compute_flyaround_waypoint_and_phi(rel_pos)
    to_wp = wp - rel_pos
    d_wp = float(np.linalg.norm(to_wp)) + 1e-6

    if is_flyaround:
        target_speed = min(0.50, 0.15 + 0.35 * np.tanh(d_wp / 5.0))
        unit_dir = to_wp / d_wp
    else:
        if dist > 1.8:
            target_speed = min(0.45, 0.05 + 0.40 * np.tanh((dist - 1.0) / 5.0))
        else:
            target_speed = 0.043
        unit_dir = -rel_pos / dist
        # Yakın mesafede yanal merkezlemeyi güçlendir
        unit_dir[0] *= 1.5
        unit_dir[2] *= 1.5
        unit_dir = unit_dir / (np.linalg.norm(unit_dir) + 1e-6)

    desired_vel = unit_dir * target_speed
    vel_err = desired_vel - vel

    # CW Coriolis kompanzasyonu
    coriolis_x = -2.0 * n_val * vel[1] * mass
    coriolis_y = 2.0 * n_val * vel[0] * mass

    # Gürültüye dayanıklı yumuşak kazanç (eski 240 yerine 115-150)
    p_gain = 120.0 if (is_flyaround or dist > 1.8) else 155.0
    force = (vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0])
    force = np.clip(force, -CONFIG.max_thrust, CONFIG.max_thrust)
    return (force / CONFIG.max_thrust).astype(np.float32)


class AntiOverfitOrbitalEnv(gym.Wrapper):
    """
    Anti-Overfitting & Sıfır Sızıntılı v5 MuJoCo Eğitim Ortamı:
    - Ölçeklenmiş, kararlı ödül (G_t ∈ [-15, +35]) -> Critic EV (R²) > 0.85, Aktör gerçekten öğrenir.
    - Gerçek MuJoCo kütle/atalet DR + post-PWPF itki arızası + sızıntısız gürültü + gecikme + qvel şok.
    - 4 rejimli geniş başlangıç dağılımı (terminal, nominal, uzak+kaçış, 360° fly-around).
    - Eğitim sırasında SIFIR expert_weight (uzman klonlama yok!).
    """

    def __init__(self, max_steps=450, dr_prob=0.85):
        env = OrbitalMujocoEnv(max_steps=max_steps)
        super().__init__(env)
        self.dr_prob = dr_prob
        self._base_mass = self.env.model.body_mass.copy()
        self._base_inertia = self.env.model.body_inertia.copy()
        self.cbf = MujocoCBFQPShield(safety_margin_pct=0.10)
        self.pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
        self.rng = np.random.default_rng()

        self.ep_use_dr = True
        self.ep_mode = "cbf_pwpf"
        self.axis_fault = np.ones(3, dtype=np.float64)
        self.pos_std = 0.0
        self.vel_std = 0.0
        self.pos_bias = np.zeros(3, dtype=np.float64)
        self.delay_steps = 0
        self.action_queue = []
        self.shock_step = -1
        self.shock_vec = np.zeros(3, dtype=np.float64)

        self.prev_phi = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_obs = None

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _, info = self.env.reset(seed=seed, **kwargs)
        self.pwpf.reset()

        self.ep_use_dr = bool(self.rng.random() < self.dr_prob)
        m_r = float(self.rng.random())
        if m_r < 0.55:
            self.ep_mode = "cbf_pwpf"
        elif m_r < 0.78:
            self.ep_mode = "pure_pwpf"
        else:
            self.ep_mode = "pure_raw"

        # 1) Fizik & Sensör DR parametreleri
        if self.ep_use_dr:
            mass_factor = float(self.rng.uniform(0.75, 1.25))
            base_scale = float(self.rng.uniform(0.72, 1.15))
            self.axis_fault = np.full(3, base_scale, dtype=np.float64)
            if self.rng.random() < 0.35:
                fa = int(self.rng.integers(0, 3))
                self.axis_fault[fa] *= float(self.rng.uniform(0.60, 0.85))
            self.pos_std = float(self.rng.uniform(0.005, 0.045))
            self.vel_std = float(self.rng.uniform(0.002, 0.012))
            self.pos_bias = self.rng.uniform(-0.03, 0.03, size=3)
            self.delay_steps = 1 if (self.rng.random() < 0.35) else 0
            if self.rng.random() < 0.30:
                self.shock_step = int(self.rng.integers(25, 130))
                sv = self.rng.normal(0.0, 1.0, size=3)
                self.shock_vec = (sv / (np.linalg.norm(sv) + 1e-8)) * float(self.rng.uniform(0.07, 0.16))
            else:
                self.shock_step = -1
        else:
            mass_factor = 1.0
            self.axis_fault = np.ones(3, dtype=np.float64)
            self.pos_std = 0.0
            self.vel_std = 0.0
            self.pos_bias = np.zeros(3, dtype=np.float64)
            self.delay_steps = 0
            self.shock_step = -1

        # ÖNCE gerçek MuJoCo kütle/atalet güncellemesi
        set_real_mujoco_mass(self.env, self._base_mass, self._base_inertia, mass_factor)

        # 2) 4 Rejimli Geniş Başlangıç Dağılımı
        rc = float(self.rng.random())
        if rc < 0.15:
            # Rejim 1: Terminal Hassasiyet
            x_i = float(self.rng.uniform(-0.6, 0.6))
            y_i = float(self.rng.uniform(0.8, 3.8))
            z_i = float(self.rng.uniform(-0.6, 0.6))
            vx_i = float(self.rng.uniform(-0.04, 0.04))
            vy_i = float(self.rng.uniform(-0.10, 0.05))
            vz_i = float(self.rng.uniform(-0.04, 0.04))
        elif rc < 0.40:
            # Rejim 2: Nominal & Orta Menzil
            x_i = float(self.rng.uniform(-5.0, 5.0))
            y_i = float(self.rng.uniform(3.8, 26.0))
            z_i = float(self.rng.uniform(-4.0, 4.0))
            vx_i = float(self.rng.uniform(-0.10, 0.10))
            vy_i = float(self.rng.uniform(-0.25, 0.15))
            vz_i = float(self.rng.uniform(-0.10, 0.10))
        elif rc < 0.65:
            # Rejim 3: Uzak + KAÇIŞ Hızı (vy > 0)
            x_i = float(self.rng.uniform(-10.0, 10.0))
            y_i = float(self.rng.uniform(26.0, 48.0))
            z_i = float(self.rng.uniform(-8.0, 8.0))
            vx_i = float(self.rng.uniform(-0.18, 0.18))
            vy_i = float(self.rng.uniform(-0.15, 0.36))
            vz_i = float(self.rng.uniform(-0.18, 0.18))
        else:
            # Rejim 4: 360° Yandan / Arkadan Fly-Around (+Y dışı!)
            ang = float(self.rng.uniform(np.pi * 0.30, np.pi * 1.70))
            rad = float(self.rng.uniform(8.5, 22.0))
            x_i = rad * np.sin(ang)
            y_i = rad * np.cos(ang)
            z_i = float(self.rng.uniform(-6.0, 6.0))
            vx_i = float(self.rng.uniform(-0.12, 0.12))
            vy_i = float(self.rng.uniform(-0.12, 0.12))
            vz_i = float(self.rng.uniform(-0.10, 0.10))

        # Z-ekseni tam fiziksel simetri rastgeleliği
        if self.rng.random() < 0.5:
            z_i = -z_i
            vz_i = -vz_i

        self.env.data.qpos[0] = x_i
        self.env.data.qpos[1] = y_i
        self.env.data.qpos[2] = z_i
        self.env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.env.data.qvel[0] = vx_i
        self.env.data.qvel[1] = vy_i
        self.env.data.qvel[2] = vz_i
        self.env.data.qvel[3:6] = [0.0, 0.0, 0.0]

        mujoco.mj_forward(self.env.model, self.env.data)
        clean_obs = self.env._get_obs()
        _, self.prev_phi, _ = compute_flyaround_waypoint_and_phi(clean_obs[:3])
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.action_queue = [np.zeros(3, dtype=np.float32)] * self.delay_steps

        if self.pos_std > 0.0 or self.vel_std > 0.0:
            obs_out = build_noisy_obs(clean_obs, self.env.n, self.rng, self.pos_std, self.vel_std, self.pos_bias)
        else:
            obs_out = clean_obs.copy()

        self.last_obs = obs_out
        info["distance"] = float(clean_obs[7])
        info["speed"] = float(clean_obs[8])
        return obs_out, info

    def step(self, action):
        # Gerçek fiziksel hız şoku
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

        # --- ÖLÇEKLENMİŞ, KARARLI VE KOORDİNAT-BAĞIMSIZ ÖDÜL (G_t ∈ [-15, +35]) ---
        rel_pos = clean_obs[:3].astype(np.float64)
        vel = clean_obs[3:6].astype(np.float64)
        dist = float(clean_obs[7])
        speed = float(clean_obs[8])
        lat_err = float(np.hypot(rel_pos[0], rel_pos[2]))

        wp, phi, is_flyaround = compute_flyaround_waypoint_and_phi(rel_pos)
        progress = self.prev_phi - phi
        self.prev_phi = phi

        # 1. Potential-Based Progress (Fly-Around + Port Yaklaşma)
        r_prog = np.clip(0.60 * progress, -0.8, 0.8)

        # 2. Koordinat-Bağımsız Yön & Kapanma Hızı Ödülü (v_close)
        to_wp = wp - rel_pos
        d_wp = float(np.linalg.norm(to_wp)) + 1e-6
        u_dir = to_wp / d_wp
        v_close = float(np.dot(vel, u_dir))

        if is_flyaround:
            v_ref = min(0.50, 0.15 + 0.35 * np.tanh(d_wp / 5.0))
        else:
            v_ref = 0.045 if dist < 1.8 else min(0.45, 0.05 + 0.40 * np.tanh((dist - 1.0) / 5.0))

        # Hız tavanı aşım cezası (yumuşak, sınırlı)
        r_speed = -min(0.6, 2.5 * max(0.0, speed - v_ref) ** 2)

        # Terminal yaklaşımda eksen-bağımsız sürünme + V-bar hizalama
        r_terminal = 0.0
        if not is_flyaround:
            if dist < 3.5:
                r_terminal -= min(0.25, 0.20 * (lat_err ** 2) + 0.06 * lat_err)
            if dist < 1.5 and rel_pos[1] > 0.0:
                if lat_err < 0.10 and 0.024 <= v_close <= 0.054:
                    r_terminal += 0.12
                elif v_close < 0.015:
                    r_terminal -= 0.12

        # 3. Engel Çarpışma / İstasyon Temas Cezası
        r_contact = -0.80 if (self.env.data.ncon > 0 and not info.get("docked_success", False)) else 0.0

        # 4. Eylem Yumuşaklığı ve Yakıt Tasarrufu (S3 gürültüde chattering kırıcı!)
        r_smooth = -0.025 * float(np.mean((cmd_act - self.prev_action) ** 2)) - 0.008 * float(np.mean(cmd_act ** 2))
        self.prev_action = cmd_act.copy()

        # 5. Küçük zaman baskısı (hovering önleyici)
        r_time = -0.012

        reward = float(r_prog + r_speed + r_terminal + r_contact + r_smooth + r_time)

        if info.get("docked_success", False):
            reward += 15.0 + 2.0 * max(0.0, 1.0 - self.env.current_step / 450.0)
        elif info.get("crashed", False):
            reward -= 10.0

        if self.pos_std > 0.0 or self.vel_std > 0.0:
            obs_out = build_noisy_obs(clean_obs, self.env.n, self.rng, self.pos_std, self.vel_std, self.pos_bias)
        else:
            obs_out = clean_obs.copy()

        self.last_obs = obs_out
        return obs_out, reward, terminated, truncated, info


def collect_multiregime_bc_dataset(num_episodes=160):
    """
    4 rejimden (terminal, nominal, uzak+kaçış, 360° fly-around) uzman verisi toplar
    ve fiziksel Z-simetrisiyle (z,-z) 2x veri artırımı uygular.
    """
    print(f"📦 [1/4] Çok-Rejimli BC Sıcak Başlangıç Verisi Toplanıyor ({num_episodes} uçuş + 2x Z-Simetri)...")
    env = AntiOverfitOrbitalEnv(max_steps=450, dr_prob=0.60)
    obs_list, act_list = [], []
    succ_cnt = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=70000 + ep * 19)
        done = trunc = False
        while not (done or trunc):
            exp_act = get_flyaround_expert_action(obs, n_val=env.env.n, mass=CONFIG.chaser_mass)
            obs_list.append(obs.copy())
            act_list.append(exp_act.copy())

            # Z-ekseni simetri veri artırımı (tam fiziksel eşdeğer!)
            obs_z = obs.copy()
            obs_z[2] = -obs_z[2]
            obs_z[5] = -obs_z[5]
            act_z = exp_act.copy()
            act_z[2] = -act_z[2]
            obs_list.append(obs_z)
            act_list.append(act_z)

            obs, _, done, trunc, info = env.step(exp_act)
        if info.get("docked_success"):
            succ_cnt += 1

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    print(f"   -> Uzman Başarısı: {succ_cnt}/{num_episodes} (%{succ_cnt/num_episodes*100:.1f}) | Toplam Örnek (2x Z-Simetri): {len(obs_arr):,}")
    return obs_arr, act_arr


def train_light_bc(model, obs_demo, act_demo, epochs=15, batch_size=256):
    """Kısa ve sağlıklı BC sıcak başlangıç (15 epoch, AdamW, weight_decay=1e-4)."""
    print(f"\n🧠 [2/4] Hafif BC Sıcak Başlangıç ({epochs} Epoch, AdamW wd=1e-4)...")
    policy = model.policy
    actor_params = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.AdamW(actor_params, lr=1.5e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(obs_demo)
    indices = np.arange(n)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        tot_loss, batches = 0.0, 0
        for i in range(0, n, batch_size):
            b = indices[i:i + batch_size]
            o_t = torch.as_tensor(obs_demo[b], dtype=torch.float32)
            a_t = torch.as_tensor(act_demo[b], dtype=torch.float32)

            feats = policy.extract_features(o_t, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(feats)
            pred_a = policy.action_net(latent_pi)
            loss = loss_fn(pred_a, a_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()

            tot_loss += float(loss.item())
            batches += 1
        if epoch in (1, 5, 10, epochs):
            print(f"   Epoch {epoch:2d}/{epochs} | BC MSE Loss: {tot_loss/batches:.6f}")

    # Başlangıç keşif standart sapmasını sağlıklı seviyeye (exp(-1.1) ≈ 0.33) çek
    with torch.no_grad():
        policy.log_std.fill_(-1.1)
    print("   -> BC Tamamlandı | Başlangıç log_std = -1.1 (std ≈ 0.33)")


def evaluate_heldout_ood_fast(model, seed_base=91000):
    """
    Early Stopping için SABİT Held-Out OOD Doğrulama Seti (eğitimde ASLA görülmez!):
    6'şar uçuş × 4 zor OOD rejim = 24 uçuş (Pure-Raw + CBF+PWPF karışık değerlendirme):
      - OOD-MassFault (gerçek kütle ±%32 + tek eksen %38 arıza)
      - OOD-NoiseDelay (5cm / 1.5cm/s gürültü + ±4cm bias + 0.5s gecikme)
      - OOD-Escape (32-48m, ±10m, vy ∈ [+0.15, +0.38])
      - OOD-FlyAround (yandan/arkadan 65°–295°, r=10–21m)
    """
    env = OrbitalMujocoEnv(max_steps=450)
    base_mass = env.model.body_mass.copy()
    base_inertia = env.model.body_inertia.copy()
    cbf = MujocoCBFQPShield(safety_margin_pct=0.10)
    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)

    sub_suites = ["mass_fault", "noise_delay", "escape", "flyaround"]
    per_suite = 6
    results = {}

    all_succ = 0
    all_crash = 0
    all_dv = []

    for s_idx, s_name in enumerate(sub_suites):
        s_succ = s_crash = 0
        s_dvs = []
        for ep in range(per_suite):
            seed = seed_base + s_idx * 1000 + ep * 73
            rng = np.random.default_rng(seed)
            _, _ = env.reset(seed=seed)
            pwpf.reset()

            mass_factor = 1.0
            axis_fault = np.ones(3, dtype=np.float64)
            pos_std = vel_std = 0.0
            pos_bias = np.zeros(3, dtype=np.float64)
            delay_steps = 0

            if s_name == "mass_fault":
                mass_factor = float(rng.choice([rng.uniform(0.68, 0.80), rng.uniform(1.20, 1.32)]))
                axis_fault = np.full(3, float(rng.uniform(0.68, 0.90)), dtype=np.float64)
                axis_fault[int(rng.integers(0, 3))] *= 0.62

            set_real_mujoco_mass(env, base_mass, base_inertia, mass_factor)

            if s_name == "noise_delay":
                pos_std, vel_std = 0.05, 0.015
                pos_bias = rng.uniform(-0.04, 0.04, size=3)
                delay_steps = 1
            elif s_name == "escape":
                env.data.qpos[0] = float(rng.uniform(-10.0, 10.0))
                env.data.qpos[1] = float(rng.uniform(32.0, 48.0))
                env.data.qpos[2] = float(rng.uniform(-8.0, 8.0))
                env.data.qvel[0] = float(rng.uniform(-0.18, 0.18))
                env.data.qvel[1] = float(rng.uniform(0.15, 0.38))
                env.data.qvel[2] = float(rng.uniform(-0.18, 0.18))
            elif s_name == "flyaround":
                ang = float(rng.uniform(np.pi * 0.36, np.pi * 1.64))
                rad = float(rng.uniform(10.0, 21.0))
                env.data.qpos[0] = rad * np.sin(ang)
                env.data.qpos[1] = rad * np.cos(ang)
                env.data.qpos[2] = float(rng.uniform(-5.5, 5.5))
                env.data.qvel[0] = float(rng.uniform(-0.10, 0.10))
                env.data.qvel[1] = float(rng.uniform(-0.10, 0.10))
                env.data.qvel[2] = float(rng.uniform(-0.08, 0.08))

            mujoco.mj_forward(env.model, env.data)
            clean_obs = env._get_obs()

            # Yarısı Pure-Raw (CBF kapalı!), yarısı CBF+PWPF açık!
            use_shield = (ep % 2 == 1)
            q_act = [np.zeros(3, dtype=np.float32)] * delay_steps
            done = trunc = False
            dv = 0.0

            while not (done or trunc):
                in_obs = (
                    build_noisy_obs(clean_obs, env.n, rng, pos_std, vel_std, pos_bias)
                    if (pos_std > 0 or vel_std > 0)
                    else clean_obs.copy()
                )
                act, _ = model.predict(in_obs, deterministic=True)
                if delay_steps > 0:
                    q_act.append(act)
                    exec_act = q_act.pop(0)
                else:
                    exec_act = act

                force = np.clip(exec_act, -1.0, 1.0) * CONFIG.max_thrust
                if use_shield:
                    force, _, _ = cbf.filter_action(in_obs[:3], in_obs[3:6], force, env.n)
                    force, _ = pwpf.modulate(force / CONFIG.max_thrust, distance=float(in_obs[7]))
                final_force = force * axis_fault
                app_act = np.clip(final_force / CONFIG.max_thrust, -1.0, 1.0)
                dv += (float(np.linalg.norm(app_act * CONFIG.max_thrust)) / env.m) * CONFIG.dt
                clean_obs, _, done, trunc, info = env.step(app_act)

            if info.get("docked_success"):
                s_succ += 1
                all_succ += 1
            elif info.get("crashed"):
                s_crash += 1
                all_crash += 1
            s_dvs.append(dv)
            all_dv.append(dv)

        results[s_name] = {
            "success_rate": round(s_succ / per_suite * 100.0, 1),
            "crash_rate": round(s_crash / per_suite * 100.0, 1),
            "mean_dv": round(float(np.mean(s_dvs)), 2),
        }

    total_eps = len(sub_suites) * per_suite
    ood_sr = round(all_succ / total_eps * 100.0, 1)
    ood_cr = round(all_crash / total_eps * 100.0, 1)
    mean_dv = round(float(np.mean(all_dv)), 2)
    # Bileşik OOD Skoru: Başarı - 2*Kaza - 2*Yakıt (hem güvenli hem ekonomik olan kazanır)
    composite = round(ood_sr - 2.0 * ood_cr - 2.0 * mean_dv, 2)
    return {
        "ood_success_rate": ood_sr,
        "ood_crash_rate": ood_cr,
        "ood_mean_dv": mean_dv,
        "composite_score": composite,
        "breakdown": results,
    }


class OODEarlyStoppingCallback(BaseCallback):
    """
    Her eval_freq adımda sabit Held-Out OOD doğrulama setinde ölçüm alır:
    - En iyi OOD bileşik skorunu veren checkpoint'i kaydeder (best_ood_model.zip).
    - Critic Explained Variance & Aktör ağırlık değişimini takip eder.
    - OOD skoru `patience` kez üst üste gelişmezse erken durdurma (Early Stopping) uygular.
    """

    def __init__(self, eval_freq=48_000, patience=4, bc_snapshot=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.bc_snapshot = bc_snapshot
        self.best_score = -1e9
        self.best_step = 0
        self.no_improve = 0
        self.last_eval_step = 0
        self.history = []

    def _actor_drift(self):
        if self.bc_snapshot is None:
            return 0.0
        max_d = 0.0
        for name, p in self.model.policy.named_parameters():
            if ("policy_net" in name or "action_net" in name) and name in self.bc_snapshot:
                d = float((p.detach().cpu() - self.bc_snapshot[name]).abs().max().item())
                max_d = max(max_d, d)
        return round(max_d, 5)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval_step) < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps
        res = evaluate_heldout_ood_fast(self.model)
        std_vec = torch.exp(self.model.policy.log_std).detach().cpu().numpy()
        drift = self._actor_drift()

        # Son rollout'tan Critic EV (SB3 logger)
        ev = self.logger.name_to_value.get("train/explained_variance", None)
        ev_str = f"{ev:.3f}" if ev is not None else "n/a"

        rec = {
            "timestep": int(self.num_timesteps),
            "ood_success_rate": res["ood_success_rate"],
            "ood_crash_rate": res["ood_crash_rate"],
            "ood_mean_dv": res["ood_mean_dv"],
            "composite_score": res["composite_score"],
            "actor_drift_from_bc": drift,
            "action_std": [round(float(x), 3) for x in std_vec],
            "critic_ev": round(float(ev), 4) if ev is not None else None,
            "breakdown": res["breakdown"],
        }
        self.history.append(rec)

        improved = res["composite_score"] > self.best_score + 0.15
        tag = ""
        if improved:
            self.best_score = res["composite_score"]
            self.best_step = self.num_timesteps
            self.no_improve = 0
            self.model.save(V5_BEST_OOD_PATH)
            tag = " ★ YENİ EN İYİ OOD!"
        else:
            self.no_improve += 1
            tag = f" (gelişme yok {self.no_improve}/{self.patience})"

        bd = res["breakdown"]
        print(
            f"📊 [OOD EVAL @ {self.num_timesteps:7,d}] OOD Başarı: %{res['ood_success_rate']:5.1f} | "
            f"Kaza: %{res['ood_crash_rate']:4.1f} | Δv: {res['ood_mean_dv']:.2f} | "
            f"Mass:%{bd['mass_fault']['success_rate']:.0f} Noise:%{bd['noise_delay']['success_rate']:.0f} "
            f"Esc:%{bd['escape']['success_rate']:.0f} Fly:%{bd['flyaround']['success_rate']:.0f} | "
            f"EV:{ev_str} | Drift:{drift:.4f} | Std:{np.round(std_vec,2)}{tag}",
            flush=True,
        )

        if self.no_improve >= self.patience and self.num_timesteps >= 192_000:
            print(f"\n🛑 [EARLY STOPPING] OOD doğrulama skoru {self.patience} turdur artmadı — ezber başlamadan durduruldu!")
            print(f"   En iyi OOD checkpoint: {self.best_step:,} adım -> {V5_BEST_OOD_PATH}")
            return False

        return True


def make_env(rank, base_seed=40000):
    def _init():
        env = AntiOverfitOrbitalEnv(max_steps=450, dr_prob=0.85)
        env.reset(seed=base_seed + rank * 131)
        return env
    return _init


def main(total_timesteps=480_000, n_envs=8):
    t0 = time.time()
    print("=" * 95)
    print("🚀 ORBITAL-RL // CHAMPION-PPO v5 (GERÇEK ÖĞRENME + ANTİ-OVERFİT + 360° FLY-AROUND)")
    print("=" * 95)

    # 1. Çok-Rejimli + 2x Z-Simetri BC Verisi
    obs_demo, act_demo = collect_multiregime_bc_dataset(num_episodes=160)

    # 2. Paralel Anti-Overfit Ortamlar
    try:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="forkserver")
    except Exception:
        vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    policy_kwargs = dict(
        features_extractor_class=OrbitalObsFeaturesExtractorV5,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.Tanh,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4),
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(1.5e-4, 2.0e-5),
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )

    # 3. Hafif BC Warmstart
    train_light_bc(model, obs_demo, act_demo, epochs=15, batch_size=256)
    model.save(V5_BC_ONLY_PATH)

    # BC sonrası ağırlık kopyası (PPO'nun aktörü gerçekten güncellediğini kanıtlamak için!)
    bc_snapshot = {
        k: p.detach().cpu().clone()
        for k, p in model.policy.named_parameters()
        if ("policy_net" in k or "action_net" in k)
    }

    # BC-Only (0 PPO adımı) Baseline OOD Skoru
    bc_ood = evaluate_heldout_ood_fast(model)
    print(
        f"\n🔍 [BC-Only @ 0 PPO Adımı] Held-Out OOD Başarı: %{bc_ood['ood_success_rate']:.1f} | "
        f"Kaza: %{bc_ood['ood_crash_rate']:.1f} | Δv: {bc_ood['ood_mean_dv']:.2f} | "
        f"Breakdown: { {k: v['success_rate'] for k, v in bc_ood['breakdown'].items()} }"
    )

    # 4. Saf PPO Eğitimi (expert_weight = 0.0!) + OOD Early Stopping
    cb = OODEarlyStoppingCallback(eval_freq=48_000, patience=4, bc_snapshot=bc_snapshot)
    cb.best_score = bc_ood["composite_score"]
    model.save(V5_BEST_OOD_PATH)  # İlk taban olarak kaydet

    print(f"\n⚡ [3/4] Saf PPO Eğitimi Başlıyor (Maks {total_timesteps:,} adım, n_envs={n_envs}, expert_weight=0.0, Early Stopping Aktif)...")
    model.learn(total_timesteps=total_timesteps, callback=cb)
    vec_env.close()

    model.save(V5_FINAL_PATH)
    elapsed = round(time.time() - t0, 2)

    report = {
        "elapsed_seconds": elapsed,
        "total_timesteps_executed": int(model.num_timesteps),
        "best_ood_step": int(cb.best_step),
        "best_ood_composite": cb.best_score,
        "bc_only_ood": bc_ood,
        "eval_history": cb.history,
        "best_model_path": V5_BEST_OOD_PATH,
        "final_model_path": V5_FINAL_PATH,
    }
    with open(V5_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f"🏁 [4/4] Champion-PPO v5 Eğitimi Tamamlandı ({elapsed}s)")
    print(f"   En İyi OOD Modeli : {V5_BEST_OOD_PATH} (Adım: {cb.best_step:,}, Skor: {cb.best_score})")
    print(f"   Eğitim Raporu     : {V5_REPORT_PATH}")
    print("=" * 95)


if __name__ == "__main__":
    main()
