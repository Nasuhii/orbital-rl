"""
Orbital-RL: Merkezi Konfigürasyon Dosyası (Single Source of Truth)
-----------------------------------------------------------------
Tüm ortamların (Pure CW Gymnasium ve MuJoCo Gymnasium) fiziksel
parametreleri, kısıtları, zaman adımları ve başarı eşikleri bu
ortak modülden beslenir.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrbitalConfig:
    # 1. Zaman Adımları ve Simülasyon Ölçeği
    dt: float = 0.5                     # Gymnasium adım süresi (saniye)
    mujoco_substep_dt: float = 0.1      # MuJoCo integratör adım süresi (s)
    mujoco_frame_skip: int = 5          # 5 * 0.1s = 0.5s (Gymnasium dt ile tam eşleşir)
    max_steps: int = 300                # 300 adım * 0.5s = 150 saniye görev bütçesi

    # 2. Uzay Aracı Kütlesi ve Eyleyici Limitleri
    chaser_mass: float = 500.0          # kg (Kovalayıcı servis aracı kütlesi)
    max_thrust: float = 15.0            # Newton (RCS 3 eksen itki tavanı: ±15 N)

    # 3. Kenetlenme ve Güvenlik Başarı Kriterleri (MANDATORY STANDARDS)
    docking_radius: float = 0.15        # Metre (15 cm hedef kenetlenme port toleransı)
    max_docking_speed: float = 0.06     # m/s (6.0 cm/s maksimum yapısal temas hızı)
    docking_cone_angle_deg: float = 25.0# Derece (LOS koridor açısı)
    max_radial_departure: float = 60.0  # Metre (Yörüngeden savrulma/kaza limiti)

    # 4. Yörüngesel Astrodinamik Parametreleri (400 km LEO / ISS İrtifası)
    orbit_altitude: float = 400_000.0   # Metre
    earth_radius: float = 6_378_137.0   # Metre
    earth_mu: float = 3.986004418e14    # m^3 / s^2

    # 5. Başlangıç Koşulları Zarfı
    init_along_track_min: float = 18.0  # Along-track (Y) başlangıç alt sınır (m)
    init_along_track_max: float = 25.0  # Along-track (Y) başlangıç üst sınır (m)
    init_radial_span: float = 2.5       # Radial (X) ± sapma (m)
    init_crosstrack_span: float = 1.8   # Cross-track (Z) ± sapma (m)
    init_speed_closing_min: float = 0.10# Hedefe doğru başlangıç hızı (m/s)
    init_speed_closing_max: float = 0.22# Hedefe doğru başlangıç hızı (m/s)

    @property
    def orbit_radius(self) -> float:
        return self.earth_radius + self.orbit_altitude

    @property
    def mean_motion(self) -> float:
        """Ortalama Yörünge Hareketi (n = sqrt(mu / r^3)) rad/s"""
        return float(np.sqrt(self.earth_mu / (self.orbit_radius ** 3)))


# Global tekil konfigürasyon örneği
CONFIG = OrbitalConfig()
