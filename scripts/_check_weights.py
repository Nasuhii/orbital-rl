import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
m3 = PPO.load(os.path.join(base, "ppo_mujoco_champion_v3_backup.zip"), device="cpu")
m4 = PPO.load(os.path.join(base, "ppo_mujoco_champion_v4.zip"), device="cpu")
mg = PPO.load(os.path.join(base, "ppo_mujoco_champion_generalized.zip"), device="cpu")

for name, m in [("v3_700k", m3), ("v4_1.4m", m4), ("gen_2.4m", mg)]:
    test_obs = torch.tensor([[0.0, 15.0, 0.0, 0.0, -0.2, 0.0, 0.0, 15.0, 0.2],
                             [0.0, 1.0, 0.0, 0.0, -0.04, 0.0, 0.0, 1.0, 0.04]], dtype=torch.float32)
    with torch.no_grad():
        v = m.policy.predict_values(test_obs).numpy().ravel()
        a, _ = m.predict(test_obs.numpy(), deterministic=True)
    print(f"{name:10s} | V(15m)={v[0]:10.2f}, V(1m)={v[1]:10.2f} | act(15m)={np.round(a[0],4)} | act(1m)={np.round(a[1],4)}")

def diff(ma, mb, tag):
    tot = 0.0
    for (k, pa), (_, pb) in zip(ma.policy.named_parameters(), mb.policy.named_parameters()):
        d = float((pa - pb).abs().max().item())
        tot = max(tot, d)
        if d > 1e-4:
            print(f"  {tag} | {k:35s} max_abs_diff={d:.6f}")
    print(f"==> {tag} overall max_abs_diff = {tot:.6f}")

diff(m3, m4, "v3 vs v4")
diff(m4, mg, "v4 vs gen")
