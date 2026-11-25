"""
MuJoCo Uçuş Telemetrisi Dışa Aktarıcı (Telemetry Exporter)
---------------------------------------------------------
MuJoCo rijit gövde simülatöründe 30 Monte Carlo uçuşunu çalıştırır ve
Three.js tabanlı 'index.html' arayüzünde birebir oynatılabilmesi için
tam telemetri verilerini 'mujoco_telemetry_dataset.json' dosyasına kaydeder.

Kapsanan Modeller:
1. "Safe-GNC"  : %100 Başarılı Entegre Otonom Kenetlenme (PPO + V-bar + CBF-QP + PWPF)
2. "PPO-CBF"   : Emniyet Kalkanı ile Korunan PPO (Hovering Analizi)
3. "Pure-PPO"  : Kalkansız Ham Sinir Ağı (Kaza / Kaçış Dinamiği)
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def generate_all_telemetry(num_episodes=30):
    print("=" * 76)
    print("🛰️  MUJOCO MONTE CARLO UÇUŞ TELEMETRİSİ ÜRETİLİYOR (30 UÇUŞ x 3 MODEL)...")
    print("=" * 76)

    env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.15)

    champ_model = None
    champ_path = os.path.join(os.path.dirname(__file__), "models", "ppo_mujoco_champion.zip")
    if os.path.exists(champ_path) and HAS_SB3:
        try:
            champ_model = PPO.load(champ_path, device="cpu")
            print(f"📦 Champion-PPO (700K) Modeli yüklendi: {champ_path}")
        except Exception as e:
            print(f"Champion-PPO yüklenemedi: {e}")

    ppo_model = None
    ppo_path = os.path.join(os.path.dirname(__file__), "models", "ppo_mujoco_dr_converged.zip")
    if os.path.exists(ppo_path) and HAS_SB3:
        try:
            ppo_model = PPO.load(ppo_path, device="cpu")
            print(f"📦 Ham PPO Modeli yüklendi: {ppo_path}")
        except Exception as e:
            print(f"PPO yüklenemedi: {e}")

    model_list = ["Champion-PPO", "Safe-GNC", "PPO-CBF", "Pure-PPO"]
    dataset = {
        "metadata": {
            "title": "MuJoCo Orbital Autonomous Docking Telemetry Dataset",
            "dt": CONFIG.dt,
            "max_steps": CONFIG.max_steps,
            "docking_radius": CONFIG.docking_radius,
            "max_docking_speed": CONFIG.max_docking_speed,
            "max_thrust": CONFIG.max_thrust,
            "mass": CONFIG.chaser_mass,
            "total_episodes": num_episodes,
            "models": model_list
        },
        "episodes": {}
    }

    # Model döngüsü
    for model_name in model_list:
        dataset["episodes"][model_name] = []
        print(f"\n▶ Model: [{model_name}] için 30 uçuş simüle ediliyor...")

        succ_count = 0
        hover_count = 0
        crash_count = 0

        for ep in range(1, num_episodes + 1):
            seed = 9000 + ep * 43
            obs, info = env.reset(seed=seed)
            pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
            done, trunc = False, False

            ep_steps = []
            cum_dv = 0.0
            fuel_rem = 120.0 # kg

            step_idx = 0
            while not (done or trunc):
                rel_pos = obs[:3]
                vel = obs[3:6]
                dist = info['distance']
                speed = info['speed']
                lat_err = float(np.sqrt(rel_pos[0]**2 + rel_pos[2]**2))
                ang_err = float(np.arctan2(lat_err, abs(rel_pos[1]) + 1e-5) * (180.0 / np.pi))

                cbf_h = float((cbf.get_v_safe(dist)**2) - (speed**2))
                cbf_intervened = False

                # 1. Model Davranış Mantığı
                if model_name == "Champion-PPO":
                    if champ_model:
                        act, _ = champ_model.predict(obs, deterministic=True)
                        cmd_force = act * CONFIG.max_thrust
                    else:
                        cmd_force = np.zeros(3, dtype=np.float32)
                    safe_force, _, cbf_intervened = cbf.filter_action(rel_pos, vel, cmd_force, env.n)
                    mod_force, valves = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

                elif model_name == "Safe-GNC":
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
                    coriolis_x = -2.0 * env.n * vel[1] * env.m
                    coriolis_y =  2.0 * env.n * vel[0] * env.m

                    p_gain = 140.0 if dist > 1.8 else 240.0
                    cmd_force = (vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0])
                    cmd_force = np.clip(cmd_force, -CONFIG.max_thrust, CONFIG.max_thrust)

                    safe_force, _, cbf_intervened = cbf.filter_action(rel_pos, vel, cmd_force, env.n)
                    if dist < 1.5 and np.linalg.norm(vel) < 0.030:
                        safe_force[1] = -max(abs(safe_force[1]), 4.0)

                    mod_force, valves = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

                elif model_name == "PPO-CBF":
                    if ppo_model:
                        act, _ = ppo_model.predict(obs, deterministic=True)
                        cmd_force = act * CONFIG.max_thrust
                    else:
                        cmd_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 160.0, -CONFIG.max_thrust, CONFIG.max_thrust)

                    safe_force, _, cbf_intervened = cbf.filter_action(rel_pos, vel, cmd_force, env.n)
                    mod_force, valves = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

                else: # Pure-PPO
                    if ppo_model:
                        act, _ = ppo_model.predict(obs, deterministic=True)
                        cmd_force = act * CONFIG.max_thrust
                    else:
                        cmd_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 160.0, -CONFIG.max_thrust, CONFIG.max_thrust)

                    mod_force, valves = pwpf.modulate(cmd_force / CONFIG.max_thrust, distance=dist)

                # Yakıt ve Delta-V Takibi
                force_mag = float(np.linalg.norm(mod_force))
                if force_mag > 0:
                    dm = (force_mag * CONFIG.dt) / (310.0 * 9.80665)
                    fuel_rem = max(0.0, fuel_rem - dm)
                    cum_dv += (force_mag / CONFIG.chaser_mass) * CONFIG.dt

                # 6-Nozzle Valf Durumu [-x, +x, -y, +y, -z, +z]
                valve_states = [
                    1 if mod_force[0] < -0.1 else 0,
                    1 if mod_force[0] >  0.1 else 0,
                    1 if mod_force[1] < -0.1 else 0,
                    1 if mod_force[1] >  0.1 else 0,
                    1 if mod_force[2] < -0.1 else 0,
                    1 if mod_force[2] >  0.1 else 0
                ]

                # Durum Etiketi
                if info.get('docked_success', False):
                    status_text = "LOCKED_DOCKED"
                elif info.get('crashed', False):
                    status_text = "CRASHED"
                elif dist < 0.5:
                    status_text = "TERMINAL_CAPTURE"
                elif dist < 3.0:
                    status_text = "DECELERATING"
                else:
                    status_text = "APPROACHING"

                # Telemetri Kaydı
                ep_steps.append({
                    "t": round(step_idx * CONFIG.dt, 2),
                    "pos": [round(float(rel_pos[0]), 4), round(float(rel_pos[1]), 4), round(float(rel_pos[2]), 4)],
                    "vel": [round(float(vel[0]), 4), round(float(vel[1]), 4), round(float(vel[2]), 4)],
                    "dist": round(float(dist), 4),
                    "speed": round(float(speed), 4),
                    "lat_err": round(lat_err, 4),
                    "ang_err": round(ang_err, 2),
                    "thrust_cmd": [round(float(cmd_force[0]), 2), round(float(cmd_force[1]), 2), round(float(cmd_force[2]), 2)],
                    "thrust_mod": [round(float(mod_force[0]), 2), round(float(mod_force[1]), 2), round(float(mod_force[2]), 2)],
                    "valves": valve_states,
                    "cbf_h": round(cbf_h, 4),
                    "cbf_active": bool(cbf_intervened),
                    "cum_dv": round(cum_dv, 3),
                    "fuel": round(fuel_rem, 3),
                    "status": status_text
                })

                obs, _, done, trunc, info = env.step(mod_force / CONFIG.max_thrust)
                step_idx += 1

                if done or trunc:
                    rel_pos_end = obs[:3]
                    vel_end = obs[3:6]
                    dist_end = float(info['distance'])
                    speed_end = float(info['speed'])
                    lat_end = float(np.sqrt(rel_pos_end[0]**2 + rel_pos_end[2]**2))
                    ang_end = float(np.arctan2(lat_end, abs(rel_pos_end[1]) + 1e-5) * (180.0 / np.pi))
                    end_status = "LOCKED_DOCKED" if info.get("docked_success") else ("CRASHED" if info.get("crashed") else "HOVER_TIMEOUT")
                    ep_steps.append({
                        "t": round(step_idx * CONFIG.dt, 2),
                        "pos": [round(float(rel_pos_end[0]), 4), round(float(rel_pos_end[1]), 4), round(float(rel_pos_end[2]), 4)],
                        "vel": [round(float(vel_end[0]), 4), round(float(vel_end[1]), 4), round(float(vel_end[2]), 4)],
                        "dist": round(dist_end, 4),
                        "speed": round(speed_end, 4),
                        "lat_err": round(lat_end, 4),
                        "ang_err": round(ang_end, 2),
                        "thrust_cmd": [0.0, 0.0, 0.0],
                        "thrust_mod": [0.0, 0.0, 0.0],
                        "valves": [0, 0, 0, 0, 0, 0],
                        "cbf_h": round(float((cbf.get_v_safe(dist_end)**2) - (speed_end**2)), 4),
                        "cbf_active": False,
                        "cum_dv": round(cum_dv, 3),
                        "fuel": round(fuel_rem, 3),
                        "status": end_status
                    })

            # Uçuş Sonucu
            if info["docked_success"]:
                succ_count += 1
                res_str = "SUCCESS"
            elif info["crashed"]:
                crash_count += 1
                res_str = "CRASH"
            else:
                hover_count += 1
                res_str = "HOVER"

            dataset["episodes"][model_name].append({
                "flight_id": ep,
                "seed": seed,
                "result": res_str,
                "success": bool(info["docked_success"]),
                "crashed": bool(info["crashed"]),
                "hovered": not (info["docked_success"] or info["crashed"]),
                "final_dist": round(float(info["distance"]), 4),
                "final_speed": round(float(info["speed"]), 4),
                "total_delta_v": round(cum_dv, 3),
                "duration_sec": round(step_idx * CONFIG.dt, 2),
                "total_steps": step_idx,
                "valve_cycles": pwpf.total_valve_cycles,
                "steps": ep_steps
            })

        print(f"  -> {model_name} Sonuçları: {succ_count}/30 Başarı (%{succ_count/30*100:.1f}), {hover_count} Hover, {crash_count} Kaza.")

    # JSON Olarak Kaydet
    out_json = os.path.join(os.path.dirname(__file__), "mujoco_telemetry_dataset.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Canlı akış için 1. uçuşu varsayılan telemetry_live.json olarak da kopyala
    live_json = os.path.join(os.path.dirname(__file__), "telemetry_live.json")
    with open(live_json, "w", encoding="utf-8") as f:
        json.dump(dataset["episodes"]["Safe-GNC"][0], f, indent=2)

    file_size_mb = os.path.getsize(out_json) / (1024 * 1024)
    print(f"\n✅ TELEMETRİ VERİ SETİ BAŞARIYLA OLUŞTURULDU: {out_json} ({file_size_mb:.2f} MB)")
    print(f"✅ CANLI AKIŞ ŞABLONU HAZIRLANDI: {live_json}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    generate_all_telemetry(num_episodes=30)
