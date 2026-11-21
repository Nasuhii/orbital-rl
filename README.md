# orbital-rl

3D autonomous spacecraft orbital docking environment and Safe-GNC reinforcement learning pipeline using Clohessy-Wiltshire relative motion dynamics, MuJoCo validation, and asymptotic reward shaping.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

- **Physics**: 3D Clohessy-Wiltshire (Hill) equations in the LVLH frame with RK4 integration, plus MuJoCo physical simulation (`orbital_docking.xml`).
- **Control & Safety**: PPO policy paired with a Control Barrier Function QP safety shield (`cbf_qp_shield.py`) and PWPF thruster modulator (`pwpf_modulator.py`).
- **Reward Shaping**: Asymptotic proximity funnel and velocity governor designed to eliminate hovering near the docking port while enforcing safe terminal contact velocity.
- **Anti-Overfitting & Verification**: Domain randomization (`train_ppo_mujoco_dr.py`), 10k cross-environment evaluation (`run_cross_env_10k.py`), and 30-minute Monte Carlo stress testing (`run_monte_carlo_30min.py`).

---

## Core Files

| File | Purpose |
| :--- | :--- |
| `orbital_env.py` / `orbital_mujoco_env.py` | Gymnasium 3D CW & MuJoCo docking environments |
| `reward_shaping.py` | Asymptotic proximity & velocity-governed reward |
| `cbf_qp_shield.py` / `pwpf_modulator.py` | Safe-GNC CBF-QP filter and thruster modulation |
| `train_mujoco_champion_v5.py` | Main v5 anti-overfit PPO training pipeline |
| `run_cross_env_10k.py` | Cross-environment generalization benchmark |
| `mujoco_live_server.py` / `index.html` | Real-time 3D telemetry visualization |

---

## Quick Start

```bash
pip install gymnasium stable-baselines3 mujoco numpy matplotlib
python3 train_mujoco_champion_v5.py
python3 run_cross_env_10k.py
```

---

## License

[MIT](LICENSE)
