"""
Gelişmiş MuJoCo PPO Eğitimi (Uzman Ön-Eğitimi + Domain Randomization + Shielded PPO)
-----------------------------------------------------------------------------------
Hedef: MuJoCo rijit gövde simülatöründe %100'e yakın başarıyla kenetlenen
saf / hibrit RL sinir ağı modeli üretmek.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


class ExpertGuidedMujocoEnv(gym.Wrapper):
    """
    MuJoCo RL Eğitim Ortamı:
    - PBRS (Potential-Based Reward Shaping) ile sıfır hovering
    - Kademeli Asimptotik Hız Profili
    - Uzman Davranış Rehberliği
    """
    def __init__(self, env, expert_guidance_weight=0.6):
        super().__init__(env)
        self.cbf = MujocoCBFQPShield(safety_margin_pct=0.15)
        self.expert_weight = expert_guidance_weight
        self.prev_dist = None
        self.prev_action = np.zeros(3, dtype=np.float32)

    def get_expert_action(self, obs):
        rel_pos = obs[:3]
        vel = obs[3:6]
        dist = float(obs[7])

        if dist > 1.8:
            target_speed = min(0.40, 0.05 + 0.35 * np.tanh((dist - 1.0) / 4.5))
        else:
            target_speed = 0.042

        unit_dir = -rel_pos / (dist + 1e-5)
        unit_dir[0] *= 1.6
        unit_dir[2] *= 1.6
        unit_dir = unit_dir / np.linalg.norm(unit_dir)
        desired_vel = unit_dir * target_speed

        vel_err = desired_vel - vel
        coriolis_x = -2.0 * self.env.n * vel[1] * self.env.m
        coriolis_y =  2.0 * self.env.n * vel[0] * self.env.m

        p_gain = 140.0 if dist > 1.8 else 240.0
        nom_force = (vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0])
        nom_force = np.clip(nom_force, -CONFIG.max_thrust, CONFIG.max_thrust)

        safe_force, _, _ = self.cbf.filter_action(rel_pos, vel, nom_force, self.env.n)
        if dist < 1.5 and np.linalg.norm(vel) < 0.030:
            safe_force[1] = -max(abs(safe_force[1]), 4.0)

        return np.clip(safe_force / CONFIG.max_thrust, -1.0, 1.0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = float(obs[7])
        self.prev_action = np.zeros(3, dtype=np.float32)
        return obs, info

    def step(self, action):
        expert_act = self.get_expert_action(self.env._get_obs())
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        dist = float(obs[7])
        speed = float(obs[8])

        # 1. Potential-Based Progress (Anti-Hovering)
        progress = (self.prev_dist - dist)
        self.prev_dist = dist
        r_progress = 60.0 * progress
        if dist < 1.5:
            r_progress += 20.0 * progress  # Son yaklaşmada ilerlemeye ekstra itici güç

        # 2. V-bar Kenetlenme Koridoru ve Yanal Sapma Cezası (X ve Z'yi sıfıra kitle)
        lat_err_sq = obs[0]**2 + obs[2]**2
        r_lateral = -30.0 * lat_err_sq
        corridor = 5.0 * max(0.0, -obs[1] / (dist + 1e-4))

        # 3. Kademeli Hız Tavanı (Son metrede durmayı değil 4.0 cm/s sürünmeyi korur)
        v_target = 0.042 if dist < 1.8 else min(0.40, 0.05 + 0.35 * np.tanh((dist - 1.0) / 4.5))
        r_speed = -35.0 * max(0.0, speed - v_target) ** 2
        # Eğer hız çok düşerse (hovering tehlikesi) hafif ceza
        if dist > 0.15 and dist < 1.5 and speed < 0.025:
            r_speed -= 10.0

        # 4. Uzman Rehberliği Hatası Cezası (Öğrenmeyi hızlandırır)
        act_diff = np.sum((action - expert_act) ** 2)
        r_expert = - self.expert_weight * 12.0 * act_diff

        # 5. Eylem Yumuşaklığı (Chattering önleme)
        r_smooth = - 0.5 * np.sum((action - self.prev_action) ** 2)
        self.prev_action = action.copy()

        total_reward = r_progress + r_lateral + corridor + r_speed + r_expert + r_smooth

        if info.get("docked_success", False):
            total_reward += 500.0
        elif info.get("crashed", False):
            total_reward -= 250.0

        return obs, float(total_reward), terminated, truncated, info


def collect_expert_demonstrations(num_episodes=40):
    """Uzman kontrolcüden durum-eylem çiftlerini toplar."""
    env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    expert_wrapper = ExpertGuidedMujocoEnv(env)

    observations = []
    actions = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=5000 + ep)
        done, trunc = False, False
        while not (done or trunc):
            exp_act = expert_wrapper.get_expert_action(obs)
            observations.append(obs.copy())
            actions.append(exp_act.copy())
            obs, _, done, trunc, _ = env.step(exp_act)

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


def pretrain_actor_with_bc(model, observations, actions, epochs=15, batch_size=128):
    """PPO Politika Ağına Imitation Learning (Behavioral Cloning) ile sıcak başlangıç yapar."""
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset_size = len(observations)
    indices = np.arange(dataset_size)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i + batch_size]
            obs_tensor = torch.as_tensor(observations[batch_idx], device="cpu")
            act_tensor = torch.as_tensor(actions[batch_idx], device="cpu")

            # Politikanın eylem tahminini hesapla
            pred_actions = policy.actor(obs_tensor) if hasattr(policy, "actor") else policy.action_net(policy.mlp_extractor.forward_actor(obs_tensor))

            loss = loss_fn(pred_actions, act_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [BC Ön-Eğitim] Epoch {epoch+1:2d}/{epochs} | MSE Loss: {total_loss/batches:.5f}")


def train_and_evaluate():
    print("=" * 76)
    print("🚀 MUJOCO YÜKSEK PERFORMANSLI PPO MODELİ EĞİTİMİ (BC + RL)")
    print("=" * 76)

    # 1. Uzman Gösterileri Toplama
    print("1. Uzmandan (Safe-GNC) 40 Uçuşluk Durum-Eylem Verisi Toplanıyor...")
    obs_demo, act_demo = collect_expert_demonstrations(num_episodes=40)
    print(f"   -> Toplanan Örnek Sayısı: {len(obs_demo)} adım.")

    # 2. PPO Modeli Başlatma
    raw_env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    guided_env = ExpertGuidedMujocoEnv(raw_env, expert_guidance_weight=0.5)
    env = Monitor(guided_env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.001,
        device="cpu",
        verbose=0
    )

    # 3. Behavioral Cloning Ön-Eğitimi
    print("\n2. PPO Aktör Ağına Sıcak Başlangıç (Behavioral Cloning)...")
    pretrain_actor_with_bc(model, obs_demo, act_demo, epochs=20)

    # 4. MuJoCo Ortamında PPO Pekiştirmeli Öğrenme İnce Ayarı
    print("\n3. MuJoCo'da PPO İnce Ayarı (75,000 Adım)...")
    model.learn(total_timesteps=75000)

    # Modeli Kaydet
    save_path = os.path.join(os.path.dirname(__file__), "models", "ppo_mujoco_high_perf.zip")
    model.save(save_path)
    print(f"\n✅ Eğitilmiş Yüksek Başarımlı PPO Modeli Kaydedildi: {save_path}")

    # 5. Doğrulama: 30 Uçuşluk Monte Carlo Testi
    print("\n" + "=" * 76)
    print("📊 4. MUJOCO DOĞRULAMA TESTİ: EĞİTİLMİŞ MODELİN 30 UÇUŞLUK MONTE CARLO PERFORMANSI")
    print("=" * 76)

    eval_env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.15)

    succ, hover, crash = 0, 0, 0
    final_dists = []
    final_speeds = []
    dvs = []

    for ep in range(1, 31):
        seed = 1000 + ep * 31
        obs, info = eval_env.reset(seed=seed)
        pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
        done, trunc = False, False
        thrust_hist = []

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            force = action * CONFIG.max_thrust

            # CBF-QP Emniyet Süzgeci
            safe_force, _, _ = cbf.filter_action(obs[:3], obs[3:6], force, eval_env.n)
            
            # Son 1.5 metrede PWPF ölü bölgesini aşmak için sürünme itkisi
            if info['distance'] < 1.5 and np.linalg.norm(obs[3:6]) < 0.030:
                safe_force[1] = -max(abs(safe_force[1]), 4.0)

            mod_force, _ = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=info['distance'])
            thrust_hist.append(mod_force)

            obs, _, done, trunc, info = eval_env.step(mod_force / CONFIG.max_thrust)

        dv = np.sum(np.linalg.norm(thrust_hist, axis=1)) / CONFIG.chaser_mass * CONFIG.dt
        final_dists.append(info['distance'])
        final_speeds.append(info['speed'])
        dvs.append(dv)

        if info['docked_success']:
            succ += 1
        elif info['crashed']:
            crash += 1
        else:
            hover += 1

    print(f"Başarı Oranı   : %{(succ/30)*100:.1f} ({succ}/30)")
    print(f"Hovering Oranı : %{(hover/30)*100:.1f} ({hover}/30)")
    print(f"Kaza Oranı     : %{(crash/30)*100:.1f} ({crash}/30)")
    print(f"Ort. Son Mesafe: {np.mean(final_dists):.3f} m (Hedef <= 0.150 m)")
    print(f"Ort. Son Hız   : {np.mean(final_speeds):.4f} m/s (Hedef <= 0.060 m/s)")
    print(f"Ort. Delta-V   : {np.mean(dvs):.2f} m/s")
    print("=" * 76)


if __name__ == "__main__":
    train_and_evaluate()
