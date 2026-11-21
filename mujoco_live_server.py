"""
Orbital-RL: Canlı MuJoCo Görsel Yayın ve Görev Kontrol Sunucusu (Live MuJoCo Visual Server)
=============================================================================================
Bu sunucu, MuJoCo fizik motorunu arka planda gerçek zamanlı (20-30 Hz) çalıştırır;
offscreen renderer ile 4 farklı kameradan canlı görüntü alır ve web tarayıcısına
yüksek performanslı, sıfır gecikmeli MJPEG ve JSON telemetri akışı sağlar.

Desteklenen Kameralar:
1. 'chase': 3. Şahıs Takip Kamerası (Dünya eğriliği ve LEO arka planı)
2. 'probe': FPV Burun Kenetlenme Probu Kamerası (Optik nişangah ve HUD)
3. 'cctv': İstasyon İskele Güvenlik Kamerası (Yaklaşan kovalayıcı görünümü)
4. 'tactical': Geniş Açılı Yörünge Taktik İzometrik Kamerası
"""

import os
import sys
import io
import time
import json
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image

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

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
PORT = 8055


class SimulationEngine:
    """Arka planda MuJoCo fizik motorunu ve kamera renderlarını işleten çekirdek"""

    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = True
        self.is_paused = False
        self.step_once_requested = False
        self.model_mode = "Champion-PPO"
        self.flight_counter = 0
        self.current_seed = 9043
        self.target_speed_override = None
        self.shock_requested = False
        self.ep_cbf_count = 0
        self.session_counts = {"docked": 0, "crashed": 0, "hover": 0}
        self.flight_history = []

        # MuJoCo Ortamı
        self.env = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
        self.cbf = MujocoCBFQPShield(safety_margin_pct=0.15)
        self.pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)

        # MuJoCo Çoklu Kamera Rendererları (HD 1280x720 + 4x MSAA)
        self.renderer_chase = OrbitalMujocoEnv.metadata.get("r_chase", None)
        self.renderer = {}
        for cam in ['chase_cam', 'probe_cam', 'station_cctv', 'orbital_tactical']:
            import mujoco
            self.renderer[cam] = mujoco.Renderer(self.env.model, 720, 1280)

        now = time.time()
        self.cam_last_seen = {
            "chase": now,
            "probe": now,
            "cctv": 0.0,
            "tactical": 0.0,
        }

        # Eğitilmiş Modeller Önbelleği
        self.loaded_models = {}
        self._load_rl_models()

        # Canlı Kare ve Telemetri Önbelleği
        self.latest_frames = {
            "chase": b"",
            "probe": b"",
            "cctv": b"",
            "tactical": b""
        }
        self.latest_telemetry = {}
        self.phase_history = []

    def _load_rl_models(self):
        if not HAS_SB3:
            return
        model_files = {
            "Champion-PPO": os.path.join(MODELS_DIR, "ppo_mujoco_champion.zip"),
            "Champion-PPO-v5": os.path.join(MODELS_DIR, "ppo_mujoco_champion_v5_best_ood.zip"),
            "BC-PPO": os.path.join(MODELS_DIR, "ppo_mujoco_high_perf.zip"),
            "DR-PPO": os.path.join(MODELS_DIR, "ppo_mujoco_dr_converged.zip")
        }
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    self.loaded_models[name] = PPO.load(path, device="cpu")
                    print(f"📦 [Sunucu] PPO Modeli Yüklendi: {name}")
                except Exception as e:
                    print(f"Model yükleme hatası ({name}): {e}")

    def reset_flight(self, seed=None):
        with self.lock:
            self.flight_counter += 1
            if seed is None:
                seed = 9000 + self.flight_counter * 43
            self.current_seed = int(seed)
            self.obs, self.info = self.env.reset(seed=self.current_seed)
            self.pwpf.reset()
            self.step_idx = 0
            self.cum_dv = 0.0
            self.fuel_rem = 120.0
            self.ep_cbf_count = 0
            self.phase_history = []
            print(f"[GNC] Flight #{self.flight_counter} started | Mode: {self.model_mode} | Seed: {self.current_seed}")

    def inject_shock(self):
        with self.lock:
            self.shock_requested = True
            print("[GNC] Impulse perturbation injected (+0.16 m/s radial, +0.12 m/s cross-track)")

    def step_simulation(self):
        if self.is_paused and not self.step_once_requested:
            time.sleep(0.05)
            return
        self.step_once_requested = False

        with self.lock:
            rel_pos = self.obs[:3]
            vel = self.obs[3:6]
            dist = float(self.info['distance'])
            speed = float(self.info['speed'])
            lat_err = float(np.sqrt(rel_pos[0]**2 + rel_pos[2]**2))
            ang_err = float(np.arctan2(lat_err, abs(rel_pos[1]) + 1e-5) * (180.0 / np.pi))

            # Dış Darbe / Şok Enjeksiyonu (Gerçek MuJoCo qvel fiziğine uygulanır!)
            if self.shock_requested:
                import mujoco as _mj
                self.env.data.qvel[0] += 0.16 # Radyal fiziksel darbe
                self.env.data.qvel[2] += 0.12 # Düzlem dışı fiziksel darbe
                _mj.mj_forward(self.env.model, self.env.data)
                self.obs = self.env._get_obs()
                rel_pos = self.obs[:3]
                vel = self.obs[3:6]
                dist = float(self.obs[7])
                speed = float(self.obs[8])
                self.shock_requested = False

            cbf_h = float((self.cbf.get_v_safe(dist)**2) - (speed**2))
            cbf_intervened = False

            # Model Kontrolcüsü Kararı
            if self.model_mode == "Safe-GNC":
                if dist > 1.8:
                    target_speed = min(0.40, 0.05 + 0.35 * np.tanh((dist - 1.0) / 4.5))
                else:
                    target_speed = self.target_speed_override if self.target_speed_override else 0.042

                unit_dir = -rel_pos / (dist + 1e-5)
                unit_dir[0] *= 1.6
                unit_dir[2] *= 1.6
                unit_dir = unit_dir / np.linalg.norm(unit_dir)
                desired_vel = unit_dir * target_speed

                vel_err = desired_vel - vel
                coriolis_x = -2.0 * self.env.n * vel[1] * self.env.m
                coriolis_y =  2.0 * self.env.n * vel[0] * self.env.m
                p_gain = 140.0 if dist > 1.8 else 240.0

                cmd_force = (vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0])
                cmd_force = np.clip(cmd_force, -CONFIG.max_thrust, CONFIG.max_thrust)

                safe_force, _, cbf_intervened = self.cbf.filter_action(rel_pos, vel, cmd_force, self.env.n)
                if dist < 1.5 and np.linalg.norm(vel) < 0.030:
                    safe_force[1] = -max(abs(safe_force[1]), 4.0)

                mod_force, _ = self.pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

            elif self.model_mode == "Champion-PPO":
                # Saf 700K Champion-PPO + CBF-QP + PWPF (Analitik servo devralması YOK)
                model = self.loaded_models.get("Champion-PPO", None)
                if model:
                    action, _ = model.predict(self.obs, deterministic=True)
                    cmd_force = action * CONFIG.max_thrust
                else:
                    cmd_force = np.zeros(3, dtype=np.float32)
                safe_force, _, cbf_intervened = self.cbf.filter_action(rel_pos, vel, cmd_force, self.env.n)
                mod_force, _ = self.pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

            elif self.model_mode == "Champion-PPO-Raw":
                # Saf 700K Champion-PPO + PWPF (CBF Kalkanı YOK, Analitik servo YOK)
                model = self.loaded_models.get("Champion-PPO", None)
                if model:
                    action, _ = model.predict(self.obs, deterministic=True)
                    cmd_force = action * CONFIG.max_thrust
                else:
                    cmd_force = np.zeros(3, dtype=np.float32)
                mod_force, _ = self.pwpf.modulate(cmd_force / CONFIG.max_thrust, distance=dist)

            elif self.model_mode == "Champion-Hybrid":
                # Hibrit Mod: PPO (d > 1.5m) + Terminal Analitik Servo (d <= 1.5m) + CBF + PWPF
                model = self.loaded_models.get("Champion-PPO", None)
                if dist > 1.5 and model:
                    action, _ = model.predict(self.obs, deterministic=True)
                    cmd_force = action * CONFIG.max_thrust
                    safe_force, _, cbf_intervened = self.cbf.filter_action(rel_pos, vel, cmd_force, self.env.n)
                else:
                    target_speed = self.target_speed_override if self.target_speed_override else 0.042
                    unit_dir = -rel_pos / (dist + 1e-5)
                    unit_dir[0] *= 1.6
                    unit_dir[2] *= 1.6
                    unit_dir = unit_dir / np.linalg.norm(unit_dir)
                    desired_vel = unit_dir * target_speed
                    vel_err = desired_vel - vel
                    nom_force = np.clip((vel_err * 240.0) + np.array([-2.0 * self.env.n * vel[1] * self.env.m, 2.0 * self.env.n * vel[0] * self.env.m, 0.0]), -CONFIG.max_thrust, CONFIG.max_thrust)
                    safe_force, _, cbf_intervened = self.cbf.filter_action(rel_pos, vel, nom_force, self.env.n)
                    if dist < 1.5 and np.linalg.norm(vel) < 0.030:
                        safe_force[1] = -max(abs(safe_force[1]), 4.0)
                mod_force, _ = self.pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)

            elif self.model_mode in self.loaded_models:
                # RL Modeli (BC-PPO veya DR-PPO)
                model = self.loaded_models[self.model_mode]
                action, _ = model.predict(self.obs, deterministic=True)
                cmd_force = action * CONFIG.max_thrust
                mod_force, _ = self.pwpf.modulate(cmd_force / CONFIG.max_thrust, distance=dist)

            elif self.model_mode == "PPO-CBF":
                # PPO + CBF Kalkanı
                model = self.loaded_models.get("DR-PPO", None)
                if model:
                    action, _ = model.predict(self.obs, deterministic=True)
                    cmd_force = action * CONFIG.max_thrust
                else:
                    cmd_force = np.clip((-rel_pos / (dist + 1e-5) * 0.35 - vel) * 160.0, -CONFIG.max_thrust, CONFIG.max_thrust)
                safe_force, _, cbf_intervened = self.cbf.filter_action(rel_pos, vel, cmd_force, self.env.n)
                mod_force, _ = self.pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)
            else:
                # Klasik Naive Model
                cmd_force = np.clip((-rel_pos / (dist + 1e-5) * 0.25 - vel * 1.5) * 100.0, -CONFIG.max_thrust, CONFIG.max_thrust)
                mod_force, _ = self.pwpf.modulate(cmd_force / CONFIG.max_thrust, distance=dist)

            # Yakıt ve Eyleyici Tüketimi
            force_mag = float(np.linalg.norm(mod_force))
            if force_mag > 0:
                dm = (force_mag * CONFIG.dt) / (310.0 * 9.80665)
                self.fuel_rem = max(0.0, self.fuel_rem - dm)
                self.cum_dv += (force_mag / CONFIG.chaser_mass) * CONFIG.dt

            valve_states = [
                1 if mod_force[0] < -0.1 else 0,
                1 if mod_force[0] >  0.1 else 0,
                1 if mod_force[1] < -0.1 else 0,
                1 if mod_force[1] >  0.1 else 0,
                1 if mod_force[2] < -0.1 else 0,
                1 if mod_force[2] >  0.1 else 0
            ]

            if cbf_intervened:
                self.ep_cbf_count += 1

            # Fizik adımını uygula (telemetri ve kamera güncel adımla tam senkron olsun)
            self.obs, _, done, trunc, self.info = self.env.step(mod_force / CONFIG.max_thrust)
            self.step_idx += 1
            reset_needed = bool(done or trunc)

            rel_pos = self.obs[:3]
            vel = self.obs[3:6]
            dist = float(self.info['distance'])
            speed = float(self.info['speed'])
            lat_err = float(np.sqrt(rel_pos[0]**2 + rel_pos[2]**2))
            ang_err = float(np.arctan2(lat_err, abs(rel_pos[1]) + 1e-5) * (180.0 / np.pi))

            # Durum Tespiti (adım sonrası gerçek sonuç)
            is_docked = bool(self.info.get('docked_success', False))
            is_crashed = bool(self.info.get('crashed', False))
            is_hover = bool(trunc and not is_docked and not is_crashed)

            if reset_needed:
                if is_docked:
                    self.session_counts["docked"] += 1
                    outcome_str = "DOCKED"
                elif is_crashed:
                    self.session_counts["crashed"] += 1
                    outcome_str = "CRASHED"
                else:
                    self.session_counts["hover"] += 1
                    outcome_str = "HOVER"
                self.flight_history.append({
                    "flight_id": self.flight_counter,
                    "seed": self.current_seed,
                    "model": self.model_mode,
                    "outcome": outcome_str,
                    "dist": round(dist, 3),
                    "speed": round(speed * 100.0, 2),
                    "dv": round(self.cum_dv, 2),
                    "duration": round(self.step_idx * CONFIG.dt, 1),
                    "cbf_count": self.ep_cbf_count,
                })
                if len(self.flight_history) > 30:
                    self.flight_history.pop(0)

            if is_docked:
                status_text = "HARD_LATCHED"
            elif is_crashed:
                status_text = "CRASHED"
            elif is_hover:
                status_text = "HOVER_TIMEOUT"
            elif dist < 0.5:
                status_text = "TERMINAL_CAPTURE"
            elif dist < 3.0:
                status_text = "DECELERATING"
            else:
                status_text = "VBAR_APPROACH"

            # Faz geçmişine ekle (grafik çizimi için)
            if self.step_idx % 2 == 0 or reset_needed:
                self.phase_history.append([round(dist, 2), round(speed, 3)])
                if len(self.phase_history) > 120:
                    self.phase_history.pop(0)

            # Kameraları Render Et (Aktif kameralar her adımda HD, diğerleri 10 adımda bir)
            cam_map = {
                'chase': 'chase_cam',
                'probe': 'probe_cam',
                'cctv': 'station_cctv',
                'tactical': 'orbital_tactical'
            }
            now = time.time()
            for key, cam_name in cam_map.items():
                is_active = (now - self.cam_last_seen.get(key, 0.0)) < 5.0
                if not is_active and (self.step_idx % 10 != 0) and self.latest_frames.get(key) and not reset_needed:
                    continue
                r = self.renderer[cam_name]
                r.update_scene(self.env.data, camera=cam_name)
                rgb = r.render()
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
                self.latest_frames[key] = buf.getvalue()

            # Telemetri Paketi
            self.latest_telemetry = {
                "flight_id": self.flight_counter,
                "seed": self.current_seed,
                "model_name": self.model_mode,
                "is_paused": self.is_paused,
                "docking_radius": CONFIG.docking_radius,
                "max_docking_speed": CONFIG.max_docking_speed,
                "step": self.step_idx,
                "max_steps": CONFIG.max_steps,
                "t": round(self.step_idx * CONFIG.dt, 2),
                "pos": [round(float(rel_pos[0]), 3), round(float(rel_pos[1]), 3), round(float(rel_pos[2]), 3)],
                "vel": [round(float(vel[0]), 4), round(float(vel[1]), 4), round(float(vel[2]), 4)],
                "dist": round(dist, 3),
                "speed": round(speed, 4),
                "v_safe": round(float(self.cbf.get_v_safe(dist)), 4),
                "lat_err": round(lat_err, 3),
                "ang_err": round(ang_err, 2),
                "thrust_mod": [round(float(mod_force[0]), 1), round(float(mod_force[1]), 1), round(float(mod_force[2]), 1)],
                "valves": valve_states,
                "cbf_h": round(cbf_h, 4),
                "cbf_active": bool(cbf_intervened),
                "cbf_count": self.ep_cbf_count,
                "cum_dv": round(self.cum_dv, 2),
                "fuel": round(self.fuel_rem, 2),
                "status": status_text,
                "docked": is_docked,
                "crashed": is_crashed,
                "hover": is_hover,
                "phase_history": self.phase_history[-50:],
                "session": {
                    "docked": self.session_counts["docked"],
                    "crashed": self.session_counts["crashed"],
                    "hover": self.session_counts["hover"],
                    "history": list(reversed(self.flight_history[-12:])),
                },
            }

            # Canlı telemetri dosyasını senkronize et (Three.js index.html için)
            try:
                live_file = os.path.join(DIR_PATH, "telemetry_live.json")
                temp_file = live_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as tf:
                    json.dump(self.latest_telemetry, tf)
                os.replace(temp_file, live_file)
            except Exception:
                pass

        # Kilit dışındayken sonuç animasyonunun görünmesi için bekle ve yeniden başlat
        if reset_needed:
            time.sleep(2.4)
            self.reset_flight()

    def run_loop(self):
        self.reset_flight(seed=1042)
        while self.is_running:
            t0 = time.time()
            self.step_simulation()
            elapsed = time.time() - t0
            sleep_time = max(0.005, 0.040 - elapsed) # ~25 Hz
            time.sleep(sleep_time)


# Global Motor
ENGINE = SimulationEngine()


class LiveStreamHandler(BaseHTTPRequestHandler):
    """MJPEG ve JSON Telemetri HTTP İstek Yöneticisi"""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Ana Görev Kontrol HTML Sayfası
        if path in ["/", "/control", "/mission"]:
            html_file = os.path.join(DIR_PATH, "mujoco_mission_control.html")
            try:
                with open(html_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error: {e}")
            return

        # 1b. 3D WebGL Simülasyon Sayfası (/3d veya /index.html)
        if path in ["/3d", "/index.html"]:
            html_file = os.path.join(DIR_PATH, "index.html")
            try:
                with open(html_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error: {e}")
            return

        # 2. Canlı MJPEG Kamera Akışları (/stream/chase, /stream/probe, /stream/cctv, /stream/tactical)
        if path.startswith("/stream/"):
            cam_key = path.split("/stream/")[1].strip("/").split("?")[0]
            if cam_key not in ["chase", "probe", "cctv", "tactical"]:
                cam_key = "chase"

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                last_frame = None
                while ENGINE.is_running:
                    ENGINE.cam_last_seen[cam_key] = time.time()
                    with ENGINE.lock:
                        jpeg_bytes = ENGINE.latest_frames.get(cam_key, b"")
                    if jpeg_bytes and jpeg_bytes != last_frame:
                        boundary = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n\r\n"
                        )
                        self.wfile.write(boundary)
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                        last_frame = jpeg_bytes
                    time.sleep(0.025) # ~40 FPS max
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        # 3. Tekil JPEG Kare İstekleri (/frame/chase vb.)
        if path.startswith("/frame/"):
            cam_key = path.split("/frame/")[1].strip("/").split("?")[0]
            if cam_key not in ["chase", "probe", "cctv", "tactical"]:
                cam_key = "chase"
            ENGINE.cam_last_seen[cam_key] = time.time()
            with ENGINE.lock:
                jpeg_bytes = ENGINE.latest_frames.get(cam_key, b"")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(jpeg_bytes)
            return

        # 4. Canlı JSON Telemetri İstekleri (/telemetry)
        if path == "/telemetry":
            with ENGINE.lock:
                telemetry_copy = dict(ENGINE.latest_telemetry)
            payload = json.dumps(telemetry_copy).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
            return

        # 5. Kontrol API İstekleri (/api/control?action=...)
        if path.startswith("/api/control"):
            query = urllib.parse.parse_qs(parsed.query)
            action = query.get("action", [""])[0]

            if action == "reset":
                seed_str = query.get("seed", [""])[0]
                seed_val = int(seed_str) if seed_str.strip().isdigit() else None
                ENGINE.reset_flight(seed=seed_val)
            elif action == "pause":
                ENGINE.is_paused = not ENGINE.is_paused
            elif action == "step_once":
                ENGINE.is_paused = True
                ENGINE.step_once_requested = True
            elif action == "shock":
                ENGINE.inject_shock()
            elif action == "set_model":
                new_model = query.get("model", ["Champion-PPO"])[0]
                ENGINE.model_mode = new_model
                ENGINE.reset_flight()
            elif action == "set_speed":
                sp = float(query.get("speed", [0.042])[0])
                ENGINE.target_speed_override = sp
            elif action == "clear_history":
                with ENGINE.lock:
                    ENGINE.session_counts = {"docked": 0, "crashed": 0, "hover": 0}
                    ENGINE.flight_history = []

            resp = json.dumps({
                "status": "ok",
                "action": action,
                "model": ENGINE.model_mode,
                "is_paused": ENGINE.is_paused,
                "seed": ENGINE.current_seed,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)
            return

        # 6. Statik Dosyalar (CSS, JS, Resimler)
        file_path = os.path.join(DIR_PATH, path.lstrip("/"))
        if os.path.exists(file_path) and os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".gif": "image/gif",
                ".mp4": "video/mp4",
                ".json": "application/json"
            }
            mime = mime_types.get(ext, "application/octet-stream")
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404, "Dosya Bulunamadi")

    def log_message(self, format, *args):
        # Konsolu gereksiz loglarla doldurma
        pass


def start_server(port=PORT):
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", port), LiveStreamHandler)
    print("=" * 80)
    print(f"🛰️  ORBITAL-RL CANLI MUJOCO GÖRSEL SUNUCUSU BAŞLATILDI!")
    print(f"🌐  Görev Kontrol Paneli : http://localhost:{port}/")
    print(f"📸  Canlı FPV Kamera     : http://localhost:{port}/stream/probe")
    print(f"📸  Canlı Takip Kamerası : http://localhost:{port}/stream/chase")
    print(f"📊  Canlı Telemetri API  : http://localhost:{port}/telemetry")
    print("=" * 80)

    # Simülasyon döngüsünü arka planda başlat
    sim_thread = threading.Thread(target=ENGINE.run_loop, daemon=True)
    sim_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor...")
        ENGINE.is_running = False
        server.shutdown()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    start_server(port_arg)
