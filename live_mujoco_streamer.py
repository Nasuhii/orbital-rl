"""
Canlı MuJoCo Telemetri Yayıncısı (Live MuJoCo Streamer)
-------------------------------------------------------
Arka planda MuJoCo fizik motorunda otonom uzay aracı kenetlenmesini
kesintisiz olarak simüle eder ve 'telemetry_live.json' dosyasına canlı
olarak yazar. Web tarayıcısı (Three.js 'index.html') bu dosyayı okuyarak
canlı uçuşu 20 Hz hızında render eder.
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG

LIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry_live.json")


def run_live_stream(model_mode="Safe-GNC"):
    env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
    cbf = MujocoCBFQPShield(safety_margin_pct=0.15)
    
    flight_counter = 1
    print(f"🚀 Canlı MuJoCo Telemetri Yayıncısı Başlatıldı! Model: {model_mode}")
    print(f"📁 Canlı Telemetri Dosyası: {LIVE_FILE}")

    while True:
        seed = int(time.time()) % 100000 + flight_counter * 17
        obs, info = env.reset(seed=seed)
        pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
        
        step_idx = 0
        cum_dv = 0.0
        fuel_rem = 120.0
        done, trunc = False, False

        print(f"\n[UÇUŞ #{flight_counter}] Başlangıç Mesafesi: {info['distance']:.2f} m | Seed: {seed}")

        while not (done or trunc):
            t_start = time.time()
            rel_pos = obs[:3]
            vel = obs[3:6]
            dist = info['distance']
            speed = info['speed']
            lat_err = float(np.sqrt(rel_pos[0]**2 + rel_pos[2]**2))
            ang_err = float(np.arctan2(lat_err, abs(rel_pos[1]) + 1e-5) * (180.0 / np.pi))

            cbf_h = float((cbf.get_v_safe(dist)**2) - (speed**2))
            cbf_intervened = False

            if model_mode == "Safe-GNC":
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
            else:
                cmd_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 160.0, -CONFIG.max_thrust, CONFIG.max_thrust)
                safe_force, _, cbf_intervened = cbf.filter_action(rel_pos, vel, cmd_force, env.n)
                mod_force, valves = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

            force_mag = float(np.linalg.norm(mod_force))
            if force_mag > 0:
                dm = (force_mag * CONFIG.dt) / (310.0 * 9.80665)
                fuel_rem = max(0.0, fuel_rem - dm)
                cum_dv += (force_mag / CONFIG.chaser_mass) * CONFIG.dt

            valve_states = [
                1 if mod_force[0] < -0.1 else 0,
                1 if mod_force[0] >  0.1 else 0,
                1 if mod_force[1] < -0.1 else 0,
                1 if mod_force[1] >  0.1 else 0,
                1 if mod_force[2] < -0.1 else 0,
                1 if mod_force[2] >  0.1 else 0
            ]

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

            payload = {
                "flight_id": flight_counter,
                "model_name": model_mode,
                "step": step_idx,
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
                "valve_cycles": pwpf.total_valve_cycles,
                "status": status_text,
                "is_terminal": bool(done or trunc),
                "docked_success": bool(info.get('docked_success', False)),
                "crashed": bool(info.get('crashed', False))
            }

            # Atomik dosya yazımı
            temp_file = LIVE_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(temp_file, LIVE_FILE)

            obs, _, done, trunc, info = env.step(mod_force / CONFIG.max_thrust)
            step_idx += 1

            # 20 Hz senkronizasyon (50 ms)
            elapsed = time.time() - t_start
            sleep_time = max(0.005, 0.050 - elapsed)
            time.sleep(sleep_time)

        res_str = "KENETLENDİ 🟢" if info.get('docked_success', False) else ("KAZA 💥" if info.get('crashed', False) else "HOVERING ⏸️")
        print(f"[UÇUŞ #{flight_counter} BİTTİ] Sonuç: {res_str} | Son Mesafe: {info['distance']:.3f} m | Son Hız: {info['speed']:.4f} m/s | Delta-V: {cum_dv:.2f} m/s")
        flight_counter += 1
        time.sleep(2.0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "Safe-GNC"
    run_live_stream(mode)
