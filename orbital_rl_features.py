"""
Orbital-RL: Sabit Fiziksel Gözlem Ölçekleyici (PyTorch Features Extractor)
-----------------------------------------------------------------------
Model ağırlıklarının içine gömülür; harici VecNormalize dosyasına gerek
kalmadan hem eğitimde hem canlı MuJoCo sunucusunda tüm 9 gözlem boyutunu
dengeli [-5, +5] aralığına ölçekler.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class OrbitalObsFeaturesExtractor(BaseFeaturesExtractor):
    """[rx, ry, rz, vx, vy, vz, coriolis, dist, speed] ölçekleyici."""

    # Karakteristik fiziksel ölçekler (X/Z son 10cm hassasiyeti için güçlendirilmiş; Y=25.0m korunur)
    OBS_SCALES = [0.5, 25.0, 0.5, 0.2, 0.5, 0.2, 0.001, 25.0, 0.5]

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        scales = torch.tensor(self.OBS_SCALES, dtype=torch.float32)
        self.register_buffer("scales", scales)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        scaled = observations / self.scales
        return torch.clamp(scaled, -5.0, 5.0)


class OrbitalObsFeaturesExtractorV5(BaseFeaturesExtractor):
    """
    Anti-Overfit & Sıfır-Körlük (Zero-Saturation) Fiziksel Özellik Çıkarıcı (v5):
    - Eski [-5, +5] sert clamp (|x|,|z|>2.5m körlüğü) kaldırıldı.
    - asinh() ölçeklemesi ile hem port önünde milimetre hassasiyeti hem 50m+ tam görüş korunur.
    - 9D standart gözlemden 16 boyutlu eksen-bağımsız (unit_dir, v_close, v_tan, lat_err, behind_flag)
      fiziksel özellik vektörü + LayerNorm üretir (mujoco_live_server ile %100 drop-in uyumlu).
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=16)
        self.norm = nn.LayerNorm(16)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        rx = observations[:, 0:1]
        ry = observations[:, 1:2]
        rz = observations[:, 2:3]
        vx = observations[:, 3:4]
        vy = observations[:, 4:5]
        vz = observations[:, 5:6]
        coriolis = observations[:, 6:7]
        dist = torch.clamp(observations[:, 7:8], min=1e-4)
        speed = torch.clamp(observations[:, 8:9], min=0.0)

        # Hedef porta birim yön vektörü (3D koordinat-bağımsız)
        ux = -rx / dist
        uy = -ry / dist
        uz = -rz / dist

        # Radyal kapanma hızı ve teğetsel sürat
        v_close = vx * ux + vy * uy + vz * uz
        v_tan = torch.sqrt(torch.clamp(speed * speed - v_close * v_close, min=1e-8))
        lat_err = torch.sqrt(rx * rx + rz * rz + 1e-8)

        # Port düzleminin arkasında/yanında olma göstergesi (fly-around farkındalığı)
        behind_flag = torch.tanh((-ry + 0.5) * 1.2)

        feats = torch.cat(
            [
                torch.asinh(rx / 1.0),
                torch.asinh(ry / 1.0),
                torch.asinh(rz / 1.0),
                torch.asinh(vx / 0.1),
                torch.asinh(vy / 0.1),
                torch.asinh(vz / 0.1),
                ux,
                uy,
                uz,
                torch.asinh(v_close / 0.1),
                torch.asinh(v_tan / 0.1),
                torch.asinh(lat_err / 1.0),
                torch.asinh(coriolis / 0.001),
                torch.asinh(dist / 2.0),
                torch.asinh(speed / 0.1),
                behind_flag,
            ],
            dim=-1,
        )
        return self.norm(feats)
