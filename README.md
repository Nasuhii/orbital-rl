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
├── orbital_env.py                  # 3D CW RK4 Gymnasium ortamı
├── orbital_mujoco_env.py           # DeepMind MuJoCo 3D rijit gövde ortamı
├── orbital_docking.xml             # Hedef istasyon, kovalayıcı ve çoklu kamera MJCF modeli
├── cbf_qp_shield.py                # Control Barrier Function (CBF-QP) güvenlik kalkanı
├── pwpf_modulator.py               # Pulse-Width Pulse-Frequency Schmitt itici modülatörü
├── rl_infrastructure.py           # Merkezi RL eğitim, benchmark ve dashboard CLI motoru
├── rl_benchmark_stats.json         # 30 uçuşluk Monte Carlo tam sayısal metrik veritabanı
├── rl_ai_training_and_benchmark_dashboard.png # 4 Panelli 300 DPI AI model eğitim ve performans grafiği
├── RL_Model_Egitimi_ve_Altyapi_Raporu.md # Uçtan uca AI model eğitimi, hiperparametre ve istatistik raporu
├── generate_visual_mission_suite.py# Çoklu kamera, FPV HUD, video ve faz düzlemi üretici motor
├── generate_visual_pdf.py          # 6 Sayfalık A4 görsel mühendislik PDF raporu oluşturucu
├── Orbital_RL_Görsel_Mühendislik_Raporu.pdf # 4 MB Yüksek çözünürlüklü görsel mühendislik PDF'i
├── Orbital_RL_Görsel_Cikti_ve_Mimari_Belgesi.md # Kapsamlı görsel çıktı ve mimari dokümantasyonu
├── mujoco_multicam_docking_sequence.png # 5 Görev Kapısı x 3 Kamera Perspektifi Storyboard
├── optical_nav_sensor_hud.png      # FPV Kokpit Optik Navigasyon ve Reticle HUD Paneli
├── phase_plane_safety_corridors.png# Faz Düzlemi (v vs d) ve CBF Kalkan Grafikleri (300 DPI)
├── mission_profile_architecture.png# Uçtan Uca LEO Görev Mimarisi ve GNC Şeması (300 DPI)
├── mujoco_live_server.py           # Gerçek zamanlı 30 FPS MuJoCo offscreen video ve telemetri yayın sunucusu
├── mujoco_mission_control.html     # Canlı çoklu kamera, HUD enstrümanları ve faz düzlemi kontrol paneli
├── docking_mission_flight.mp4      # Canlı telemetrili 60 FPS tam uçuş videosu
├── docking_mission_flight.gif      # Hareketli simülasyon animasyonu
├── train_ppo.py                    # Stable-Baselines3 PPO eğitim betiği
├── evaluate_and_plot.py            # Çoklu uçuş başarı telemetrisi değerlendiricisi
└── run_monte_carlo_30_step6.py     # 30 uçuşluk Monte Carlo master doğrulama scripti
```

---

## 🎮 Canlı MuJoCo Görev Kontrol ve Görsel Telemetri Arayüzü (30 FPS)

MuJoCo fizik motorunda otonom kenetlenmeyi web tarayıcısından canlı ve sıfır gecikmeyle (30 FPS MJPEG video akışı, kokpit HUD göstergeleri, dinamik faz düzlemi ve interaktif model kontrolü ile) izlemek için:

```bash
# Canlı MuJoCo video ve telemetri sunucusunu başlatın:
python mujoco_live_server.py 8055
```

Tarayıcınızda açın:  
🌐 **`http://localhost:8055/`** — [mujoco_mission_control.html](mujoco_mission_control.html)

### 🌟 Canlı Arayüzün Özellikleri:
- **Çoklu Kamera Akışı (Multi-Cam Stream):** Chase Cam (3. şahıs LEO takibi), FPV Optical Probe Cam (burun nişangah HUD), Station CCTV ve Taktik Yörünge kameraları arasında tek tıkla anında geçiş ve PiP (Picture-in-Picture).
- **Canlı HUD Enstrümanları:** Sol Hız Bandı (Velocity Tape, 6 cm/s limit), Sağ Mesafe Bandı (Range Tape, 0.15 m port), FPV nişangah ağı ve durum banner'ı.
- **Dinamik Faz Düzlemi (v vs d):** Aracın yaklaşma rotasını HTML5 Canvas üzerinde gerçek zamanlı çizerek CBF yeşil emniyet koridorunu ve kırmızı kaza bölgesini canlı takip etme.
- **Canlı Eyleyici LED Matrisi:** PWPF Schmitt modülatörünün $\pm 15\text{ N}$ 3 eksenli itici darbelerini canlı alevlenmelerle görme.
- **İnteraktif Kontrol Deck:** Model değiştirme (`Integrated Safe-GNC`, `BC-PPO`, `DR-PPO`, `PPO-CBF`), mikrogöktaşı şoku enjekte etme ve yeni uçuş başlatma.

---

## 📸 Görsel ve Telemetri Çıktı Pipeline'ı

Simülasyonun ürettiği tüm çoklu kamera görüntülerini, kokpit HUD göstergelerini, faz düzlemi grafiklerini ve hareketli video/GIF çıktısını üretmek için:

```bash
# Master görsel görev paketini çalıştırın:
python generate_visual_mission_suite.py

# 6 Sayfalık yüksek çözünürlüklü A4 Görsel Mühendislik PDF raporunu oluşturun:
python generate_visual_pdf.py
```

Detaylı görsel rapor ve mühendislik analizleri için:  
👉 **[Orbital_RL_Görsel_Cikti_ve_Mimari_Belgesi.md](Orbital_RL_Görsel_Cikti_ve_Mimari_Belgesi.md)** 📑  
👉 **[Orbital_RL_Görsel_Mühendislik_Raporu.pdf](Orbital_RL_Görsel_Mühendislik_Raporu.pdf)** 📑

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
