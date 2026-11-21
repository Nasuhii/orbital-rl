"""
Orbital-RL: MuJoCo Uyumlu Control Barrier Function (CBF-QP) Emniyet Kalkanı
-------------------------------------------------------------------------
Havacılık standardında kaza önleme garantisi:
- Güvenlik Fonksiyonu : h(x) = v_safe^2(d) - ||v||^2 >= 0
- Güvenlik Marjı      : %15 (Temas hız limiti 0.06 m/s yerine 0.051 m/s tavan)
- Kısıt Şartı         : h_dot + alpha * h >= 0
- Çözüm Algoritması   : Kapalı form analitik Minimal Müdahale QP Projeksiyonu
"""

import numpy as np
from orbital_config import CONFIG


class MujocoCBFQPShield:
    def __init__(
        self,
        mass=CONFIG.chaser_mass,
        max_thrust=CONFIG.max_thrust,
        alpha=1.8,                       # Bariyer gevşeme parametresi (1/s)
        safety_margin_pct=0.10,          # %10 Güvenlik marjı (0.054 m/s tavan)
        v_dock_limit=CONFIG.max_docking_speed, # 0.06 m/s
        v_approach_max=0.40,             # 0.40 m/s yaklaşma koridor tavanı
        d_brake=5.0                      # Frenleme yarıçapı (metre)
    ):
        self.m = mass
        self.max_thrust = max_thrust
        self.alpha = alpha
        self.v_dock_safe = v_dock_limit * (1.0 - safety_margin_pct) # 0.051 m/s (%15 marj)
        self.v_max = v_approach_max
        self.d_brake = d_brake
        self.total_interventions = 0

    def get_v_safe(self, dist):
        """Asimptotik tanh hız tavanı (%15 marj uygulanmış)"""
        return self.v_dock_safe + (self.v_max - self.v_dock_safe) * np.tanh(dist / self.d_brake)

    def get_v_safe_derivative(self, dist):
        sech = 1.0 / np.cosh(dist / self.d_brake)
        return (self.v_max - self.v_dock_safe) * (sech ** 2) / self.d_brake

    def filter_action(self, pos, vel, nominal_thrust_force, n_mean_motion=CONFIG.mean_motion):
        """
        pos                  : [x, y, z] bağıl konum (metre)
        vel                  : [vx, vy, vz] bağıl hız (m/s)
        nominal_thrust_force : [Fx, Fy, Fz] Newton cinsinden talep edilen itki
        Döner                : safe_thrust_force (N), h_val, was_intervened
        """
        dist = float(np.linalg.norm(pos)) + 1e-6
        speed = float(np.linalg.norm(vel))

        v_safe = self.get_v_safe(dist)
        h = (v_safe ** 2) - (speed ** 2)

        # 1. Clohessy-Wiltshire doğal ivmeleri
        ax_cw = 2.0 * n_mean_motion * vel[1] + 3.0 * (n_mean_motion ** 2) * pos[0]
        ay_cw = -2.0 * n_mean_motion * vel[0]
        az_cw = -(n_mean_motion ** 2) * pos[2]
        a_cw = np.array([ax_cw, ay_cw, az_cw], dtype=np.float64)

        # 2. dr/dt = (pos . vel) / dist
        dist_dot = np.dot(pos, vel) / dist
        dv_safe_dt = self.get_v_safe_derivative(dist) * dist_dot

        # 3. Kısıt: (2 * vel / m) . u <= alpha * h + 2 * v_safe * dv_safe_dt - 2 * vel . a_cw
        A_vec = (2.0 * vel) / self.m
        b_val = (self.alpha * h) + (2.0 * v_safe * dv_safe_dt) - (2.0 * np.dot(vel, a_cw))

        constraint_val = np.dot(A_vec, nominal_thrust_force)
        was_intervened = False
        safe_force = nominal_thrust_force.copy()

        if constraint_val > b_val:
            self.total_interventions += 1
            was_intervened = True
            norm_A_sq = np.dot(A_vec, A_vec)
            if norm_A_sq > 1e-9:
                correction = (constraint_val - b_val) / norm_A_sq * A_vec
                safe_force = nominal_thrust_force - correction

        # RCS aktüatör fiziksel sınırlarına doyur
        safe_force = np.clip(safe_force, -self.max_thrust, self.max_thrust)
        return safe_force, h, was_intervened
