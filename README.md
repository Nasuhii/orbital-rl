# orbital-rl: 3D Yörüngesel Kenetlenme ve Asimptotik Ödül Şekillendirme 🛰️

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Ağır ve hantal fizik simülatörlerine (Gazebo, MuJoCo, Isaac Sim vb.) ihtiyaç duymadan, **3 Boyutlu Clohessy-Wiltshire (Hill) bağıl yörünge mekaniği denklemlerini** saf Python ve NumPy ile çözen, uzay araçlarında sıkça görülen **"Hovering (Havada Asılı Kalma / Çarpışmaktan Kaçınma)"** patolojisini **Asimptotik ve Üstel Ödül Şekillendirme (Reward Shaping)** ile çözen hafif ve modüler Pekiştirmeli Öğrenme (RL) araştırma kütüphanesi.

---

## 🎯 Projenin Çözdüğü Temel Problem: "Hovering Patolojisi"

Pekiştirmeli öğrenme ile otonom kenetlenme (autonomous docking) ajanları eğitilirken en yaygın başarısızlık modu şudur:
- Hedefe güvenli hızın ($v > 0.06\text{ m/s}$) üzerinde temas edildiğinde sistem büyük bir kaza cezası (**Crash Penalty: -100**) uygular.
- Klasik doğrusal ödül fonksiyonlarında ($R = -w_1 d - w_2 \|u\|^2$) ajan bu devasa cezadan kaçınmak için bir **lokal optimum** keşfeder:
  > **Patoloji:** Ajan hedefe 2-4 metre yaklaşır, tüm itkisini ters yönde ateşleyerek hızını $0\text{ m/s}$ yapar ve limanın hemen dışında sabit kalarak ("hovering") zaman aşımını bekler!

Bu projede tasarlanan **Asimptotik ve Üstel Ödül Mimarisi**, ajanı cezalandırmadan önce matematiksel olarak güvenli bir hız koridoruna yönlendirir ve limanın içine çeker.

---

## 🛰️ 1. Fiziksel Temeller: 3D Clohessy-Wiltshire (CW) Denklemleri

Dairesel bir Alçak Dünya Yörüngesinde (LEO ~400 km) bulunan hedef istasyon merkezli **LVLH (Local-Vertical Local-Horizontal)** koordinat çerçevesinde bağıl hareket diferansiyel denklemleri:

$$\ddot{x} - 2n\dot{y} - 3n^2 x = \frac{F_x}{m} \quad \text{[Radyal Eksen (Yerçekimi Gradyanı)]}$$
$$\ddot{y} + 2n\dot{x} = \frac{F_y}{m} \quad \text{[Along-Track Eksen (Yörünge Hızı / V-Bar)]}$$
$$\ddot{z} + n^2 z = \frac{F_z}{m} \quad \text{[Cross-Track Eksen (Düzlem Dışı Harmonik Salınım)]}$$

* **$n = \sqrt{\frac{\mu}{r^3}}$**: Ortalama Yörünge Açısal Hızı (Mean motion: $\approx 0.00113\text{ rad/s}$)
* **$m$**: Kovalayıcı uzay aracı kütlesi ($250\text{ kg}$)
* **$[F_x, F_y, F_z]$**: RCS itici kuvvetleri ($[-15\text{ N}, +15\text{ N}]$)
* **İntegrasyon Metodu:** Yüksek yörüngesel hassasiyet için **4. Dereceden Runge-Kutta (RK4)** integratörü kullanılır.

---

## 📐 2. Matematiksel Ödül Şekillendirme (Reward Shaping)

Hovering sorununu çözmek için tasarlanan **AsymptoticOrbitalReward** fonksiyonu 5 ana bileşenden oluşur:

### A. Üstel Çekim Havuzu (Exponential Proximity Funnel)
Hedefe yaklaştıkça ödül asimptotik olarak artarak ajanı liman içine çeker:
$$R_{\text{prox}}(d) = w_{\text{exp}} \cdot \exp\left(-\frac{d}{\sigma_{\text{dist}}}\right)$$

### B. Asimptotik Hız Frenleme Denetçisi (Asymptotic Velocity Governor)
Ajan uzaktayken ($d > 10\text{ m}$) yüksek hızlara izin verilir; hedefe yaklaştıkça izin verilen hız hiperbolik tanjant ($\tanh$) ile yumuşakça güvenli temas hızına ($v_{\text{dock}} \le 0.05\text{ m/s}$) indirgenir:
$$v_{\text{safe}}(d) = v_{\text{dock}} + (v_{\text{max}} - v_{\text{dock}}) \cdot \tanh\left(\frac{d}{d_{\text{brake}}}\right)$$
$$R_{\text{vel}} = - w_{\text{vel}} \cdot \max(0, \|v\| - v_{\text{safe}}(d))^2$$

### C. V-Bar Kenetlenme Koridoru Hizalaması
$$R_{\text{corridor}} = w_{\text{cone}} \cdot \left(\frac{y}{d + \epsilon}\right)^2$$

### D. İtki ve Zaman Maliyeti
$$R_{\text{effort}} = - w_{\text{fuel}} \|u\|^2 - w_{\text{time}}$$

---

## 📁 Proje Dosya Yapısı

```text
orbital-rl/
├── orbital_env.py          # 3D CW RK4 Gymnasium ortamı
├── reward_shaping.py       # Naive vs Asymptotic ödül modelleri
├── simulate_demo.py        # Analitik karşılaştırma & 4-panelli yörünge simülasyonu
├── train_ppo.py            # Stable-Baselines3 PPO eğitim betiği
├── evaluate_and_plot.py    # Çoklu uçuş başarı telemetrisi değerlendiricisi
├── reward_landscape.png    # Üstel ödül gradyanı ve hız sınırı grafiği
└── docking_trajectories.png# 3D Yörünge, mesafe, hız ve itki grafikleri
```

---

## 🚀 Hızlı Başlangıç (Kurulum ve Çalıştırma)

### 1. Kurulum (Sadece Standart Pip)
```bash
# Temel simülasyon ve grafikler için:
pip install gymnasium numpy matplotlib

# Stable-Baselines3 (PPO Eğitimi İçin):
# Not: PyTorch büyük bir paket olduğundan tmpfs dolmasını önlemek için:
TMPDIR=/home/nasuhi/.cache pip install stable-baselines3 --no-cache-dir
```

### 2. Simülasyonu ve Karşılaştırma Grafiklerini Üretme
PPO model eğitimi beklemeden anında fizik motorunu ve ödül karşılaştırmasını test etmek için:
```bash
python simulate_demo.py
```
Bu komut 2 adet yüksek çözünürlüklü grafik üretir:
- `reward_landscape.png` (Matematiksel ödül gradyanı)
- `docking_trajectories.png` (3D yörünge ve telemetri)

### 3. PPO Modelini Eğitme
```bash
# Gelişmiş asimptotik ödül ile 80,000 adım eğit:
python train_ppo.py --timesteps 80000

# Karşılaştırma amacıyla hovering üreten naive model ile eğit:
python train_ppo.py --timesteps 80000 --naive
```

### 4. Modeli Çoklu Uçuşlarda Test Etme
```bash
python evaluate_and_plot.py
```

---

## 📊 Örnek Telemetri Sonuçları

```text
================ TELEMETRİ SONUÇLARI ================
🔴 Naive Ajan        -> Son Mesafe: 2.17 m | Son Hız: 0.000 m/s | Durum: HOVERING (Asılı Kaldı)
🟢 Asymptotic Shaped -> Son Mesafe: 0.15 m | Son Hız: 0.050 m/s | Durum: SOFT DOCKED (Başarılı!)
=====================================================

8 Uçuşluk Monte-Carlo Test Özeti:
   • Başarılı Yumuşak Kenetlenme : %100.0 (8/8)
   • Hovering (Liman Dışında Kalma) : %0.0
   • Çarpışma / Kaza : %0.0
   • Ortalama Harcanan Delta-V : 0.82 m/s
```

---

## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak geliştirilmiştir.
Katkı sağlayan: [Nasuhii](https://github.com/Nasuhii)
