"""
Reward Shaping Modülleri - Orbital-RL Projesi
--------------------------------------------
Uzay aracı kenetlenmesinde (Orbital Docking) karşılaşılan en büyük problem:
"Hovering (Havada Asılı Kalma / Çarpışmaktan Kaçınma)" patolojisidir.

Klasik ödül fonksiyonlarında hedef limana çarpmamak için uygulanan büyük negatif
ceza (crash penalty), ajanı limanın hemen dışında durup sıfır hızla bekleme
(asılı kalma) lokal optimumuna kilitler.

Bu modül iki stratejiyi karşılaştırmalı olarak sunar:
1. NaiveReward: Asılı kalmaya (hovering) yol açan geleneksel doğrusal ödül.
2. AsymptoticOrbitalReward: Üstel çekim kuyusu (exponential gravity well),
   mesafeye bağlı asimptotik hız frenleme (asymptotic velocity governor)
   ve koridor hizalama ödüllerini birleştiren üst düzey ödül mimarisi.
"""

import numpy as np


class BaseRewardShaper:
    """Tüm ödül şekillendirme sınıfları için temel arayüz."""
    def compute_reward(self, state, action, prev_state, info):
        raise NotImplementedError


class NaiveReward(BaseRewardShaper):
    """
    Geleneksel Doğrusal Ödül (Hovering Patolojisi Üreten Baz Model).
    
    R = - w_pos * ||r|| - w_fuel * ||u||^2 - w_time
    Crash: -100 (Hız sınırını aşarak hedefe çarpma)
    Dock: +100
    
    Sonuç: Ajan -100 cezasından kaçmak için hedefe 2-4 metre kala durur (hovering).
    """
    def __init__(self, w_pos=0.1, w_fuel=0.01, w_time=0.05, crash_penalty=100.0, dock_bonus=100.0):
        self.w_pos = w_pos
        self.w_fuel = w_fuel
        self.w_time = w_time
        self.crash_penalty = crash_penalty
        self.dock_bonus = dock_bonus

    def compute_reward(self, state, action, prev_state, info):
        # state: [x, y, z, vx, vy, vz]
        pos = state[:3]
        vel = state[3:6]
        dist = np.linalg.norm(pos)
        speed = np.linalg.norm(vel)
        thrust_mag = np.linalg.norm(action)

        reward = - (self.w_pos * dist) - (self.w_fuel * (thrust_mag ** 2)) - self.w_time

        # Terminal Durumlar
        if info.get('docked_success', False):
            reward += self.dock_bonus
        elif info.get('crashed', False):
            reward -= self.crash_penalty

        return float(reward)


class AsymptoticOrbitalReward(BaseRewardShaper):
    """
    Üstel ve Asimptotik Ödül Şekillendirme (Hovering Çözümü).
    
    Bileşenler:
    1. Exponential Proximity Funnel: r_dist = w_exp * exp(-dist / sigma_dist)
       Hedefe yaklaştıkça ödül asimptotik olarak patlayarak ajanı içeri çeker.
    2. Dynamic Velocity Governor (Asimptotik Hız Eşleme):
       v_safe(dist) = v_dock + (v_approach - v_dock) * tanh(dist / d_brake)
       Ajan uzaktayken hızlı yaklaşıma izin verilir, yaklaştıkça v_safe hiperbolik olarak
       yanaşma hızına (v_dock <= 0.05 m/s) düşürülür. İhlaller karesel olarak cezalandırılır.
    3. Approach Corridor Alignment:
       Kenetlenme eksenine (V-bar / +y ekseni) hizalanma açısı ödüllendirilir.
    4. Anti-Hovering Momentum Bonus:
       Hedefe doğru olan radyal hız projeksiyonu ödüllendirilir.
    """
    def __init__(
        self,
        w_exp=25.0,
        sigma_dist=3.0,
        w_vel_penalty=8.0,
        v_dock_safe=0.03,          # 3 cm/s hedef temas hızı
        v_approach_max=0.5,        # 0.5 m/s yaklaşma tavanı
        d_brake=8.0,               # 8 metreden itibaren kademeli frenleme
        w_corridor=5.0,
        w_closing=4.0,
        w_fuel=0.005,
        w_time=0.08,
        dock_bonus=300.0,
        crash_penalty=150.0
    ):
        self.w_exp = w_exp
        self.sigma_dist = sigma_dist
        self.w_vel_penalty = w_vel_penalty
        self.v_dock_safe = v_dock_safe
        self.v_approach_max = v_approach_max
        self.d_brake = d_brake
        self.w_corridor = w_corridor
        self.w_closing = w_closing
        self.w_fuel = w_fuel
        self.w_time = w_time
        self.dock_bonus = dock_bonus
        self.crash_penalty = crash_penalty

    def safe_velocity_limit(self, dist):
        """Mesafeye bağlı izin verilen maksimum güvenli yanaşma hızı (Asimptotik Tanh Filtresi)."""
        return self.v_dock_safe + (self.v_approach_max - self.v_dock_safe) * np.tanh(dist / self.d_brake)

    def compute_reward(self, state, action, prev_state, info):
        pos = state[:3]
        vel = state[3:6]
        prev_pos = prev_state[:3] if prev_state is not None else pos

        dist = np.linalg.norm(pos)
        prev_dist = np.linalg.norm(prev_pos)
        speed = np.linalg.norm(vel)
        thrust_norm = np.linalg.norm(action)

        # 1. Üstel Yakınlaşma Çekim Havuzu (Exponential Proximity Funnel)
        r_proximity = self.w_exp * np.exp(-dist / self.sigma_dist)

        # 2. Asimptotik Hız Frenleme Denetçisi (Asymptotic Velocity Governor)
        v_limit = self.safe_velocity_limit(dist)
        excess_speed = max(0.0, speed - v_limit)
        r_vel = - self.w_vel_penalty * (excess_speed ** 2)

        # 3. Yaklaşma Hızı ve İlerleme Bonusu (Anti-Hovering Closing Reward)
        # Hedefe doğru atılan her adım pozitif fark üretir
        dist_progress = prev_dist - dist
        r_progress = self.w_closing * dist_progress

        # 4. Kenetlenme Koridoru / Konisi Hizalama (V-bar / +Y yaklaşım ekseni)
        # Hedef (0,0,0) noktasında ve kenetlenme portu +y yönünde kabul edilir.
        if dist > 0.05:
            # Yaklaşma vektörünün +y ekseni ile yaptığı açı kosinüsü
            corridor_alignment = max(0.0, pos[1] / dist)
            r_corridor = self.w_corridor * (corridor_alignment ** 2)
        else:
            r_corridor = self.w_corridor

        # 5. Efor ve Zaman Maliyeti (Yakıt + Zaman aşımı engelleme)
        r_effort = - (self.w_fuel * (thrust_norm ** 2)) - self.w_time

        # Toplam Şekillendirilmiş Ödül
        total_reward = r_proximity + r_vel + r_progress + r_corridor + r_effort

        # 6. Terminal Sonuçlar
        if info.get('docked_success', False):
            total_reward += self.dock_bonus
        elif info.get('crashed', False):
            total_reward -= self.crash_penalty

        return float(total_reward)
