"""
Orbital-RL: PWPF (Pulse-Width Pulse-Frequency) İtici Modülatörü & Chattering Önleyici
-----------------------------------------------------------------------------------
Uzay aracı RCS soğuk gaz valfleri oransal (analog) çalışamaz; yalnızca tam AÇIK (+15N)
veya tam KAPALI (0N) olabilir. 

Bu modül:
1. Alçak Geçiren Filtre (Km, Tm): Yüksek frekanslı sinir ağı gürültüsünü sönümler.
2. Schmitt Tetikleyici (Uon, Uoff): Histerezis eşikleriyle titreşimi (chattering) engeller.
3. Eylem Değişim Oran Sınırlayıcı (Rate Limiter): Ani itki sıçramalarını yumuşatır.
4. Yakın Mesafe İtki Ölçekleme (Proximity Thrust Governor): Son 2 metrede limit döngüsünü keser.
"""

import numpy as np


class PWPFModulator3Axis:
    def __init__(
        self,
        dt=0.5,
        Km=1.0,
        Tm=1.2,                # Filtre zaman sabiti (MuJoCo dt=0.5s için optimize)
        U_on=0.20,             # Açılma eşiği (%20 komut)
        U_off=0.08,            # Kapanma eşiği (%8 komut)
        max_thrust=15.0,
        action_rate_limit=0.40 # Adım başına izin verilen maksimum komut değişimi
    ):
        self.dt = dt
        self.Km = Km
        self.Tm = Tm
        self.U_on = U_on
        self.U_off = U_off
        self.max_thrust = max_thrust
        self.rate_limit = action_rate_limit

        self.filter_state = np.zeros(3, dtype=np.float64)
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.valve_output = np.zeros(3, dtype=np.float64)
        self.total_valve_cycles = 0  # Valf yorgunluğu sayacı

    def reset(self):
        self.filter_state[:] = 0.0
        self.prev_action[:] = 0.0
        self.valve_output[:] = 0.0
        self.total_valve_cycles = 0

    def modulate(self, continuous_cmd, distance=10.0):
        """
        continuous_cmd: [-1, 1] sürekli eylem vektörü
        distance      : Hedefe olan mesafe (metre)
        Döner         : modulated_force (Newton), valve_active (bool array)
        """
        # 1. Yakın Mesafe İtki Ölçekleme (Son 2 metrede doygunluk chattering'ini engelle)
        if distance < 2.0:
            thrust_scale = max(0.20, min(1.0, 0.20 + 0.80 * (distance / 2.0)))
        else:
            thrust_scale = 1.0

        # 2. Eylem Oran Sınırlayıcı (Slew-Rate Limiting)
        delta = continuous_cmd - self.prev_action
        clipped_delta = np.clip(delta, -self.rate_limit, self.rate_limit)
        smooth_cmd = self.prev_action + clipped_delta
        self.prev_action = smooth_cmd.copy()

        # 3. 3 Eksen Alçak Geçiren Filtre: dy/dt = (Km * u - y) / Tm
        # Euler ayrıklaştırması: y[k+1] = y[k] + (dt / Tm) * (Km * u[k] - y[k])
        alpha = min(0.85, self.dt / self.Tm)
        self.filter_state += alpha * (self.Km * smooth_cmd - self.filter_state)

        # 4. Schmitt Tetikleyici (Histerezis Açma/Kapama)
        new_output = np.zeros(3, dtype=np.float64)
        for i in range(3):
            val = self.filter_state[i]
            prev_out = self.valve_output[i]

            if abs(val) >= self.U_on:
                new_output[i] = np.sign(val)
            elif abs(val) <= self.U_off:
                new_output[i] = 0.0
            else:
                new_output[i] = prev_out # Histerezis aralığında eski durumu koru

            # Valf açılıp kapandıysa sayacı artır
            if new_output[i] != prev_out and new_output[i] != 0.0:
                self.total_valve_cycles += 1

        self.valve_output = new_output.copy()
        modulated_force = self.valve_output * (self.max_thrust * thrust_scale)
        return modulated_force, (new_output != 0.0)
