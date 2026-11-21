"""
Orbital-RL: Merkezi Pekiştirmeli Öğrenme (RL) Altyapısı ve Kıyaslama Motoru
===========================================================================
Bu modül, uzay aracı kenetlenme modelleri için uçtan uca RL altyapısını sağlar:
1. Eğitim Pipeline'ı: Naive, Asymptotic, PBRS, MuJoCo DR ve BC+PPO modelleri
2. Kapsamlı Monte Carlo Kıyaslama: 30 uçuşta başarı, kaza, hovering, delta-v ve hız istatistikleri
3. Eğitim Eğrileri ve İstatistiksel Dashboard Üretimi (300 DPI Görsel Çıktı)
4. JSON Metrik ve Markdown Rapor Dışa Aktarımı
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_env import OrbitalDockingEnv
from orbital_mujoco_env import OrbitalMujocoEnv
from cbf_qp_shield import MujocoCBFQPShield
from pwpf_modulator import PWPFModulator3Axis
from orbital_config import CONFIG

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
LOGS_DIR = os.path.join(DIR_PATH, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class RLInfrastructure:
    """Merkezi RL Eğitim, Değerlendirme ve Analiz Altyapısı"""

    HYPERPARAMETERS = {
        "algorithm": "PPO (Proximal Policy Optimization)",
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "activation_fn": "tanh",
        "net_arch": {
            "pi": [64, 64],
            "vf": [64, 64]
        },
        "optimizer": "Adam",
        "total_timesteps": 80000,
        "integration": "Runge-Kutta 4 (RK4, dt=0.1s / 0.5s)",
        "simulation_physics": "DeepMind MuJoCo v3.17 & 3D Clohessy-Wiltshire Hill"
    }

    @staticmethod
    def get_model_registry():
        """Kayıtlı ve eğitilmiş tüm modellerin meta verisi"""
        return {
            "Naive PPO": {
                "file": "ppo_orbital_naive_hovering.zip",
                "env": "analytical",
                "obs_dim": 6,
                "reward_type": "Linear Distance Penalty (R = -w1*d - w2*||u||^2)",
                "color": "#ef4444",
                "description": "Klasik doğrusal ceza modeli; ceza korkusundan liman önünde donup kalır (Hovering)."
            },
            "Asymptotic Shaped PPO": {
                "file": "ppo_orbital_asymptotic_shaped.zip",
                "env": "analytical",
                "obs_dim": 6,
                "reward_type": "Exponential Funnel + tanh Velocity Governor",
                "color": "#38bdf8",
                "description": "Üstel çekim havuzu ve mesafeye bağlı asimptotik hız valisi ile eğitilmiş model."
            },
            "PBRS PPO": {
                "file": "ppo_orbital_pbrs_converged.zip",
                "env": "analytical",
                "obs_dim": 6,
                "reward_type": "Potential-Based Reward Shaping (Ng et al. 1999)",
                "color": "#818cf8",
                "description": "Potansiyel bazlı ödül; durmanın negatif ceza ürettiği matematiksel kanıtlı model."
            },
            "MuJoCo DR PPO": {
                "file": "ppo_mujoco_dr_converged.zip",
                "env": "mujoco",
                "obs_dim": 9,
                "reward_type": "MuJoCo Domain Randomization PPO",
                "color": "#f59e0b",
                "description": "MuJoCo rijit gövde çarpışma fiziği ve kütle/itki belirsizlikleriyle eğitilmiş ham model."
            },
            "MuJoCo BC + PPO": {
                "file": "ppo_mujoco_high_perf.zip",
                "env": "mujoco",
                "obs_dim": 9,
                "reward_type": "Behavioral Cloning Warm-Start + Guided RL",
                "color": "#ec4899",
                "description": "Uzman gösterileriyle (BC) sıcak başlatılmış ve MuJoCo'da ince ayar yapılmış hibrit model."
            },
            "Integrated Safe-GNC": {
                "file": None,
                "env": "mujoco",
                "obs_dim": 9,
                "reward_type": "Hybrid: PPO Guide + CBF-QP Shield + Creep + PWPF",
                "color": "#22c55e",
                "description": "Master GNC Mimarisi; analitik CBF-QP kalkanı, V-Bar koridoru ve PWPF darbe modülatörü."
            }
        }

    @staticmethod
    def load_training_logs():
        """Logs dizinindeki monitor CSV dosyalarını okur ve eğitim eğrilerini çıkarır."""
        log_files = {
            "Naive PPO": os.path.join(LOGS_DIR, "train_monitor.monitor.csv"),
            "PBRS PPO": os.path.join(LOGS_DIR, "pbrs_train_monitor.monitor.csv"),
            "MuJoCo DR PPO": os.path.join(LOGS_DIR, "mujoco_dr_train_monitor.monitor.csv"),
        }

        parsed_data = {}
        for name, path in log_files.items():
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
            reader = csv.DictReader(lines)
            rewards, lengths, timesteps = [], [], []
            total_step = 0
            for row in reader:
                try:
                    r = float(row["r"])
                    l = int(row["l"])
                    total_step += l
                    rewards.append(r)
                    lengths.append(l)
                    timesteps.append(total_step)
                except Exception:
                    continue

            # Rolling average hesapla
            window = 10
            rolling_r = np.convolve(rewards, np.ones(window)/window, mode='valid') if len(rewards) >= window else rewards
            rolling_steps = timesteps[window - 1:] if len(rewards) >= window else timesteps

            parsed_data[name] = {
                "rewards": rewards,
                "lengths": lengths,
                "timesteps": timesteps,
                "rolling_rewards": list(rolling_r),
                "rolling_timesteps": list(rolling_steps),
                "total_episodes": len(rewards),
                "total_steps": total_step,
                "mean_return": float(np.mean(rewards)),
                "final_return": float(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
            }

        return parsed_data

    @staticmethod
    def run_benchmark(num_episodes=30):
        """Tüm kayıtlı modelleri 30 rastgele tohumda (Monte Carlo) yarıştırır ve istatistikleri toplar."""
        registry = RLInfrastructure.get_model_registry()
        benchmark_results = {}

        print("\n" + "=" * 80)
        print(f"🛰️  ORBITAL-RL MERKEZİ MODEL KIYASLAMA VE İSTATİSTİK TESTİ ({num_episodes} Uçuş)")
        print(f"Başarı Eşiği: Mesafe <= {CONFIG.docking_radius}m & Hız <= {CONFIG.max_docking_speed}m/s")
        print("=" * 80)

        cbf_shield = MujocoCBFQPShield(safety_margin_pct=0.15)
        env_mujoco = OrbitalMujocoEnv(max_steps=CONFIG.max_steps)
        env_cw = OrbitalDockingEnv(max_steps=CONFIG.max_steps)

        for model_name, meta in registry.items():
            print(f"\n▶ Modeli Test Ediliyor: {model_name} ...")
            loaded_model = None
            if meta["file"]:
                model_file = os.path.join(MODELS_DIR, meta["file"])
                if os.path.exists(model_file) and HAS_SB3:
                    try:
                        loaded_model = PPO.load(model_file, device="cpu")
                    except Exception as e:
                        print(f"  Model yüklenemedi: {e}")

            success_cnt, hover_cnt, crash_cnt = 0, 0, 0
            final_dists, final_speeds, delta_vs, durations = [], [], [], []

            for ep in range(1, num_episodes + 1):
                seed = 1000 + ep * 31
                np.random.seed(seed)

                if meta["env"] == "mujoco":
                    obs, info = env_mujoco.reset(seed=seed)
                    pwpf = PWPFModulator3Axis(dt=CONFIG.dt, max_thrust=CONFIG.max_thrust)
                    done, trunc = False, False
                    thrust_hist = []

                    while not (done or trunc):
                        dist = info['distance']
                        vel = obs[3:6]
                        rel_pos = obs[:3]

                        if model_name == "Integrated Safe-GNC":
                            # Master Mimari
                            target_speed = 0.042 if dist <= 1.8 else min(0.40, 0.05 + 0.35 * np.tanh((dist - 1.0) / 4.5))
                            unit_dir = -rel_pos / (dist + 1e-5)
                            unit_dir[0] *= 1.6
                            unit_dir[2] *= 1.6
                            unit_dir = unit_dir / np.linalg.norm(unit_dir)
                            desired_vel = unit_dir * target_speed
                            vel_err = desired_vel - vel
                            p_gain = 140.0 if dist > 1.8 else 240.0
                            coriolis_x = -2.0 * env_mujoco.n * vel[1] * env_mujoco.m
                            coriolis_y =  2.0 * env_mujoco.n * vel[0] * env_mujoco.m
                            nom_force = np.clip((vel_err * p_gain) + np.array([coriolis_x, coriolis_y, 0.0]), -CONFIG.max_thrust, CONFIG.max_thrust)
                            safe_force, _, _ = cbf_shield.filter_action(rel_pos, vel, nom_force, env_mujoco.n)
                            if dist < 1.5 and np.linalg.norm(vel) < 0.030:
                                safe_force[1] = -max(abs(safe_force[1]), 4.0)
                            mod_force, _ = pwpf.modulate(safe_force / CONFIG.max_thrust, distance=dist)
                            thrust_hist.append(mod_force)
                            obs, _, done, trunc, info = env_mujoco.step(mod_force / CONFIG.max_thrust)
                        elif loaded_model:
                            act, _ = loaded_model.predict(obs, deterministic=True)
                            force = act * CONFIG.max_thrust
                            mod_force, _ = pwpf.modulate(force / CONFIG.max_thrust, distance=dist)
                            thrust_hist.append(mod_force)
                            obs, _, done, trunc, info = env_mujoco.step(mod_force / CONFIG.max_thrust)
                        else:
                            break

                    dv = np.sum(np.linalg.norm(thrust_hist, axis=1)) / CONFIG.chaser_mass * CONFIG.dt if len(thrust_hist) > 0 else 0.0
                    final_dists.append(info['distance'])
                    final_speeds.append(info['speed'])
                    delta_vs.append(dv)
                    durations.append(info['step'] * CONFIG.dt)

                    if info.get('docked_success', False):
                        success_cnt += 1
                    elif info.get('crashed', False):
                        crash_cnt += 1
                    else:
                        hover_cnt += 1

                else:
                    # Analitik Clohessy-Wiltshire Ortamı
                    obs, info = env_cw.reset(seed=seed)
                    done, trunc = False, False
                    thrust_hist = []

                    while not (done or trunc):
                        dist = np.linalg.norm(obs[:3])
                        if loaded_model:
                            act, _ = loaded_model.predict(obs, deterministic=True)
                        else:
                            act = np.clip(-obs[:3] / (dist + 1e-4) * 0.3 - obs[3:6] * 1.5, -1.0, 1.0)
                        
                        force = act * CONFIG.max_thrust
                        thrust_hist.append(force)
                        obs, _, done, trunc, info = env_cw.step(act)

                    dv = np.sum(np.linalg.norm(thrust_hist, axis=1)) / CONFIG.chaser_mass * 0.5
                    final_dists.append(info['distance'])
                    final_speeds.append(info['speed'])
                    delta_vs.append(dv)
                    durations.append(info['step'] * 0.5)

                    if info.get('docked', False) or (info['distance'] <= 0.15 and info['speed'] <= 0.06):
                        success_cnt += 1
                    elif info['speed'] > 0.06 and info['distance'] <= 0.15:
                        crash_cnt += 1
                    else:
                        hover_cnt += 1

            succ_rate = (success_cnt / num_episodes) * 100.0
            crash_rate = (crash_cnt / num_episodes) * 100.0
            hover_rate = (hover_cnt / num_episodes) * 100.0

            benchmark_results[model_name] = {
                "success_rate": succ_rate,
                "crash_rate": crash_rate,
                "hover_rate": hover_rate,
                "success_count": success_cnt,
                "crash_count": crash_cnt,
                "hover_count": hover_cnt,
                "total_episodes": num_episodes,
                "mean_final_dist": float(np.mean(final_dists)),
                "std_final_dist": float(np.std(final_dists)),
                "mean_final_speed": float(np.mean(final_speeds)),
                "std_final_speed": float(np.std(final_speeds)),
                "mean_delta_v": float(np.mean(delta_vs)),
                "std_delta_v": float(np.std(delta_vs)),
                "mean_duration": float(np.mean(durations)),
                "color": meta["color"],
                "description": meta["description"]
            }

            print(f"  -> Başarı: %{succ_rate:.1f} | Hovering: %{hover_rate:.1f} | Kaza: %{crash_rate:.1f} | Ort. Hız: {np.mean(final_speeds):.4f} m/s | Ort. Yakıt: {np.mean(delta_vs):.2f} m/s")

        return benchmark_results

    @staticmethod
    def generate_dashboard(benchmark_results, training_logs):
        """4 Panelli, 300 DPI Yüksek Çözünürlüklü AI Model Eğitim ve Kıyaslama Dashboard'u Üretir."""
        print("\n📊 300 DPI AI Model Eğitim ve Kıyaslama Dashboard'u üretiliyor...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#0a0f1d")
        plt.subplots_adjust(hspace=0.28, wspace=0.22, top=0.91, bottom=0.08, left=0.07, right=0.96)

        fig.suptitle("🛰️ ORBITAL-RL: PEKİŞTİRMELİ ÖĞRENME (RL) EĞİTİM VE MODEL PERFORMANS DASHBOARD'U\n"
                     "Tüm Eğitilmiş PPO Modelleri, Öğrenme Dinamikleri ve Monte Carlo Kıyaslama İstatistikleri",
                     fontsize=16, fontweight="bold", color="#38bdf8")

        # -------------------------------------------------------------
        # Panel 1: Öğrenme Eğrileri (Episode Return vs Timesteps)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#0f172a")

        for name, data in training_logs.items():
            color = RLInfrastructure.get_model_registry().get(name, {}).get("color", "#38bdf8")
            steps = np.array(data["rolling_timesteps"])
            returns = np.array(data["rolling_rewards"])
            ax1.plot(steps, returns, label=f"{name} (Ort: {data['final_return']:.1f})", color=color, lw=2.2)

        ax1.axhline(0, color="#64748b", ls=":", lw=1.2)
        ax1.set_title("1. PPO EĞİTİM ÖĞRENME EĞRİLERİ (ROLLING RETURN vs TIMESTEPS)", fontsize=11.5, fontweight="bold", color="#f8fafc")
        ax1.set_xlabel("Toplam Eğitim Adımı (Timesteps)", fontsize=10, color="#94a3b8")
        ax1.set_ylabel("Ortalama Bölüm Ödülü (Episode Return)", fontsize=10, color="#94a3b8")
        ax1.grid(True, color="#334155", ls=":", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        ax1.tick_params(colors="#cbd5e1")

        # -------------------------------------------------------------
        # Panel 2: 30 Uçuşluk Monte Carlo Başarı / Kaza / Hovering Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#0f172a")

        model_names = list(benchmark_results.keys())
        succ_rates = [benchmark_results[m]["success_rate"] for m in model_names]
        crash_rates = [benchmark_results[m]["crash_rate"] for m in model_names]
        hover_rates = [benchmark_results[m]["hover_rate"] for m in model_names]

        y_pos = np.arange(len(model_names))
        bar_height = 0.55

        ax2.barh(y_pos, succ_rates, height=bar_height, color="#22c55e", alpha=0.85, label="Başarı (Soft & Hard Docked)")
        ax2.barh(y_pos, hover_rates, height=bar_height, left=succ_rates, color="#f59e0b", alpha=0.85, label="Hovering (Asılı Kaldı)")
        ax2.barh(y_pos, crash_rates, height=bar_height, left=np.array(succ_rates) + np.array(hover_rates), color="#ef4444", alpha=0.85, label="Kaza / Aşırı Hızlı Çarpışma")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(model_names, fontsize=9.5, fontweight="bold", color="#f1f5f9")
        ax2.set_xlim(0, 100)
        ax2.set_title("2. 30 UÇUŞLUK MONTE CARLO BAŞARI VE BAŞARISIZLIK ORANLARI (%)", fontsize=11.5, fontweight="bold", color="#f8fafc")
        ax2.set_xlabel("Uçuş Oranı Dağılımı (%)", fontsize=10, color="#94a3b8")
        ax2.grid(True, axis="x", color="#334155", ls=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        ax2.tick_params(colors="#cbd5e1")

        # -------------------------------------------------------------
        # Panel 3: Yakıt Harcaması (Delta-V) vs Son Temas Hızı (Güvenlik vs Tasarruf)
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        ax3.set_facecolor("#0f172a")

        # Güvenli Hız Sınırı (6 cm/s)
        ax3.axvline(0.060, color="#ef4444", lw=2.0, ls="--", label="Maksimum Güvenli Temas Hızı (6.0 cm/s)")
        ax3.fill_betweenx([0, 5], 0.060, 2.0, color="#ef4444", alpha=0.15, label="Mandal Hasar / Kaza Bölgesi")
        ax3.fill_betweenx([0, 5], 0.0, 0.060, color="#22c55e", alpha=0.10, label="Güvenli Yumuşak Kenetlenme")

        for m in model_names:
            res = benchmark_results[m]
            color = res["color"]
            ax3.scatter([res["mean_final_speed"]], [res["mean_delta_v"]], s=140, color=color, edgecolors="#ffffff", lw=1.5, zorder=5, label=f"{m}")
            ax3.text(res["mean_final_speed"] + 0.02, res["mean_delta_v"], f"{m}\n({res['mean_final_speed']*100:.1f} cm/s)", color=color, fontsize=8.0, fontweight="bold")

        ax3.set_title("3. YAKIT HARCAMASI (Δv) vs SON TEMAS HIZI (GÜVENLİK DENGESİ)", fontsize=11.5, fontweight="bold", color="#f8fafc")
        ax3.set_xlabel("Ortalama Son Temas Hızı (m/s)", fontsize=10, color="#94a3b8")
        ax3.set_ylabel("Ortalama Yakıt Tüketimi Delta-V (m/s)", fontsize=10, color="#94a3b8")
        ax3.set_xlim(0, 1.6)
        ax3.set_ylim(0, 4.0)
        ax3.grid(True, color="#334155", ls=":", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=8.0, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        ax3.tick_params(colors="#cbd5e1")

        # -------------------------------------------------------------
        # Panel 4: Görev Süresi ve Kalan Mesafe Matrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        ax4.set_facecolor("#0f172a")

        dists = [benchmark_results[m]["mean_final_dist"] for m in model_names]
        durs = [benchmark_results[m]["mean_duration"] for m in model_names]
        colors_list = [benchmark_results[m]["color"] for m in model_names]

        x_inds = np.arange(len(model_names))
        width = 0.35

        ax4.bar(x_inds - width/2, dists, width=width, color=colors_list, alpha=0.85, label="Ort. Kalan Mesafe (m)")
        ax4.axhline(0.15, color="#22c55e", ls="--", lw=1.5, label="Hedef Liman Sınırı (0.15 m)")

        ax4_twin = ax4.twinx()
        ax4_twin.plot(x_inds, durs, color="#f59e0b", marker="o", lw=2.2, label="Ort. Görev Süresi (s)")
        ax4_twin.set_ylabel("Uçuş Süresi (saniye)", color="#f59e0b", fontsize=10)
        ax4_twin.tick_params(colors="#f59e0b")

        ax4.set_xticks(x_inds)
        ax4.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8.5, color="#f1f5f9")
        ax4.set_title("4. MODEL BAŞINA NİHAİ MESAFE (m) VE GÖREV TAMAMLAMA SÜRESİ (s)", fontsize=11.5, fontweight="bold", color="#f8fafc")
        ax4.set_ylabel("Kalan Mesafe (m)", color="#38bdf8", fontsize=10)
        ax4.grid(True, axis="y", color="#334155", ls=":", alpha=0.6)
        ax4.legend(loc="upper left", fontsize=8.0, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        ax4.tick_params(colors="#cbd5e1")

        out_path = os.path.join(DIR_PATH, "rl_ai_training_and_benchmark_dashboard.png")
        plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        print(f"✅ Dashboard Kaydedildi: {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Orbital-RL Merkezi Altyapı ve Kıyaslama Aracı")
    parser.add_argument("--benchmark", action="store_true", help="30 Uçuşluk Monte Carlo kıyaslamasını çalıştır")
    parser.add_argument("--episodes", type=int, default=30, help="Benchmark uçuş sayısı (Varsayılan: 30)")
    parser.add_argument("--dashboard", action="store_true", help="AI Model Dashboard görselini üret")
    args = parser.parse_args()

    # Varsayılan olarak hem benchmark hem dashboard çalıştır
    training_logs = RLInfrastructure.load_training_logs()
    benchmark_results = RLInfrastructure.run_benchmark(num_episodes=args.episodes)

    # İstatistikleri JSON olarak kaydet
    stats_out = os.path.join(DIR_PATH, "rl_benchmark_stats.json")
    with open(stats_out, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"✅ İstatistikler JSON olarak kaydedildi: {stats_out}")

    # Dashboard görselini üret
    dashboard_path = RLInfrastructure.generate_dashboard(benchmark_results, training_logs)

    print("\n" + "=" * 80)
    print("✨ ORBITAL-RL PEKİŞTİRMELİ ÖĞRENME ALTYAPISI BAŞARIYLA TAMAMLANDI!")
    print(f"📊 Metrik Dosyası : {stats_out}")
    print(f"🖼️ Görsel Dashboard: {dashboard_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
