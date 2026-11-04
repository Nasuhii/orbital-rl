TARGET DECK: data

# 🛰️ Orbital-RL: 3D Yörüngesel Kenetlenme ve Ödül Şekillendirme Görsel Raporu

> **Özel Depo (Private Repo):** [github.com/Nasuhii/orbital-rl](https://github.com/Nasuhii/orbital-rl) 🔒  
> **📄 Akademik & Mühendislik PDF Raporu:** [[Areas/Software/orbital-rl/Orbital_RL_Proje_Raporu.pdf|Orbital_RL_Proje_Raporu.pdf]] 📑  
> **Geliştirici:** Nasuhii  
> **Fizik Modeli:** 3D Clohessy-Wiltshire (Hill) Diferansiyel Denklemleri & RK4 İntegratörü  
> **Çözülen Problem:** Havada Asılı Kalma (Hovering / Stalling) Patolojisi  

---

## 📸 1. Simülasyonun Ürettiği Görsel Grafikler

### A. 3D Yörünge, Mesafe, Hız ve İtki Paneli
Aşağıdaki görselde sol tarafta hedefe 3 metre kala korkup duran **Naive Ajan (Kırmızı)** ile sağ tarafta asimptotik ödül ile limana sıfır hatayla yanaşan **Shaped Ajan (Yeşil)** karşılaştırılmaktadır:

![3D Kenetlenme Yörüngeleri ve Telemetri](docking_trajectories.png)

#### 🔍 Grafikteki 4 Panelin Anlamı:
1. **Sol Üst (3D Yaklaşma Rotası):** Uzay aracının (Chaser) istasyona (Target $(0,0,0)$) 3 boyutta nasıl yaklaştığını gösterir. Kırmızı çizgi 3. metrede kilitlenirken, yeşil çizgi tam yıldıza (docking port) ulaşır.
2. **Sağ Üst (Mesafe - Zaman):** Kırmızı çizgi 3.2 metrede yatay bir çizgiye dönüşür (Hovering). Yeşil çizgi ise $0.15\text{ m}$ kenetlenme eşiğinin altına inerek kenetlenmeyi tamamlar.
3. **Sol Alt (Bağıl Hız Profili):** Yeşil ajanın hızının, kaza yapmamak için mesafeye bağlı olarak asimptotik şekilde frenlendiğini ve $0.05\text{ m/s}$ güvenli temas hızında limana girdiğini gösterir.
4. **Sağ Alt (RCS İtici Kuvveti):** İticilerin (Thrusters) harcadığı kuvveti (Newton) gösterir.

---

### B. Ödül Şekillendirme Matematiği (Reward Landscape)

![Ödül Şekillendirme Gradyanı ve Hız Sınırı](reward_landscape.png)

#### 🔍 Grafikteki 2 Panelin Anlamı:
1. **Sol Grafik (Ödül Gradyanı):** Kırmızı doğrusal ceza hedefe yaklaştıkça güçlü bir çekim yaratamazken; yeşil eğri ($25 \cdot e^{-d/3}$) son 3 metrede üstel olarak yükselerek ajana *"İçeri gir, büyük ödülü al!"* sinyali verir.
2. **Sağ Grafik (Asimptotik Hız Denetçisi):** Aracın hedefe olan mesafesine göre aşmaması gereken maksimum güvenli hız koridorunu ($v_{\text{safe}}(d) = 0.03 + 0.47 \cdot \tanh(d/8)$) gösterir.

---

## 📊 2. Karşılaştırmalı Telemetri Verileri

| Metrik | Naive Ajan (Geleneksel Doğrusal Ödül) | Asymptotic Shaped Ajan (Bizim Geliştirmemiz) | İdeal Güvenlik Kriteri |
| :--- | :---: | :---: | :---: |
| **Son Kalan Mesafe** | `2.17 m` 🔴 | `0.15 m` 🟢 | $\le 0.15\text{ m}$ |
| **Son Temas Hızı** | `0.000 m/s` (Sabit kaldı) | `0.050 m/s` 🟢 | $\le 0.060\text{ m/s}$ |
| **Kenetlenme Durumu** | ⏸️ **HOVERING (Asılı Kaldı)** | 🟢 **SOFT DOCKED (Başarılı)** | Başarılı Kenetlenme |
| **Başarı Oranı (8 Uçuş)** | `%0.0` | **`%100.0` (8/8 Uçuş)** | $> \%95$ |
| **Ortalama Delta-V** | `0.55 m/s` (Yakıt israfı) | `0.82 m/s` (Optimal yanaşma) | Minimum Yakıt |

---

## ⚡ 3. Ekstrem Monte Carlo Stres ve Dayanıklılık Testi (30 Uçuşluk Ağır Bozucu Testi)

Uzay araçları otonom kenetlenme (RPO) operasyonlarında NASA ve ESA standartlarına uygun olarak simüle edilen 5 ağır bozan etken:
1. **Genişletilmiş Başlangıç Zarfı:** $35 - 50\text{ m}$ mesafe (nominalin 2 katı) ve $+0.15 \dots +0.35\text{ m/s}$ istasyondan kaçış yönünde tehlikeli başlangıç rotası.
2. **LIDAR & Doppler Sensör Gürültüsü:** $\sigma_{\text{pos}} = 5\text{ cm}$ RMS konum, $\sigma_{\text{vel}} = 1\text{ cm/s}$ RMS hız ölçüm gürültüsü.
3. **Aktüatör Bozulması:** $\pm 15\%$ itki dalgalanması (jitter) ve $F_z$ (Cross-track) ekseninde $\%30$ kalıcı RCS verim kaybı.
4. **Yörüngesel Şok Darbesi ($t=50\text{ s}$):** $0.12\text{ m/s}$ anlık rastgele mikrogöktaşı / dış kuvvet darbesi.
5. **LOS Emniyet Konisi:** $\pm 30^\circ$ yaklaşma hunisi ihlallerinin denetlenmesi.

![30 Uçuşluk Monte Carlo Ekstrem Stres Testi](stress_test_results.png)

### 📊 Karşılaştırmalı Stres Testi Sonuçları (30 Monte Carlo Uçuşu)

| Değerlendirme Metriği | Asymptotic Shaped Kontrolcü (Bizim Mimari) | Naive Doğrusal Kontrolcü (Klasik Ceza) | Durum Analizi |
| :--- | :---: | :---: | :--- |
| **Kenetlenme Başarı Oranı** | **`%100.0` (30/30 Uçuş)** 🟢 | `%0.0` (0/30 Uçuş) 🔴 | Asymptotic ağır bozuculara rağmen tam kenetlendi |
| **Hovering (Takılıp Kalma)** | **`%0.0`** 🟢 | `%100.0` 🔴 | Naive ceza korkusuyla 2-3 m'de donup kaldı |
| **Çarpışma / Kaza Oranı** | **`%0.0`** 🟢 | `%0.0` 🟢 | İki model de kaza yapmadı |
| **Ortalama Delta-V (Yakıt)** | **`1.78 m/s`** 🟢 | `3.29 m/s` (İsraf) 🔴 | Asymptotic %46 daha az yakıt harcadı |
| **Ortalama Görev Süresi** | **`123.8 s`** ⚡ | `350.0 s` (Zaman Aşımı) ⏱️ | Asymptotic 3 kat daha hızlı tamamladı |
| **Ortalama Koridor İhlali** | **`0.0 adım`** 🛡️ | `0.2 adım` ⚠️ | Emniyet konisi sınırlarını kusursuz korudu |

---

## 🛠️ 4. Kod Dosyaları Ne İşe Yarar?

- **[orbital_env.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/orbital_env.py):** Uzay fiziğini (3D Clohessy-Wiltshire diferansiyel denklemleri) 4. derece Runge-Kutta (RK4) sayısal integratörü ile saniye saniye çözen Gymnasium ortamıdır.
- **[reward_shaping.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/reward_shaping.py):** Ajanın hovering yapmasını engelleyen matematiksel ödül formüllerinin bulunduğu dosyadır.
- **[simulate_demo.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/simulate_demo.py):** İki ajanı aynı başlangıç noktasından başlatıp yarıştırır ve nominal görseli üretir.
- **[stress_test_monte_carlo.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/stress_test_monte_carlo.py):** Sensör gürültüsü, itici bozulması, yörüngesel şok ve kaçış rotasını içeren 30 uçuşluk Monte Carlo dayanıklılık stres testidir.
- **[train_ppo.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/train_ppo.py):** Stable-Baselines3 kütüphanesiyle PPO yapay zeka modelini eğiten eğitim kodudur.
- **[evaluate_and_plot.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Documents/Obsidian%20Vault/Areas/Software/orbital-rl/evaluate_and_plot.py):** Farklı açılardan 8 ayrı uçuş testi yaparak başarı tablosunu ekrana basar.

---

## 🃏 Anki Kartları

START
Temel
TARGET DECK: data
Ön: Orbital-RL: Asimptotik Hız Frenleme Denetçisinde (Velocity Governor) neden hiperbolik tanjant (tanh) fonksiyonu kullanılır?
Geri: $\tanh(x)$ fonksiyonu $x \to 0$ iken doğrusal şekilde sıfıra yaklaşır, $x$ büyüdükçe ise asimptotik olarak $1$'e doyuma ulaşır.<br>Böylece uzay aracı uzaktayken sabit ve yüksek bir yaklaşma hızına izin verilirken, limana yaklaştıkça hız tavanı pürüzsüzce sıfıra indirilerek sert çarpışmalar önlenir.
Tags: from_obsidian, orbital_rl, reward_shaping, math
END

START
Temel
TARGET DECK: data
Ön: Orbital Otonom Kenetlenme (RPO) sistemlerinde Monte Carlo Stres Testi neden yapılır ve hangi bozucular test edilir?
Geri: Deterministik yörüngeler gerçek uzay ortamındaki sensör belirsizliklerini yansıtamaz.<br>1) <b>LIDAR/Doppler sensör gürültüsü</b> (5 cm, 1 cm/s),<br>2) <b>İtici bozulması & jitter</b> (RCS verim kaybı),<br>3) <b>Dış darbe/şoklar</b> (mikrogöktaşı, Coriolis sapması),<br>4) <b>Aşırı başlangıç sapmaları</b> test edilerek kontrolcünün güvenli koridorda (LOS Cone) kalıp kalamadığı ve yakıt (Delta-V) bütçesi doğrulanır.
Tags: from_obsidian, orbital_rl, monte_carlo, rpo, gnc
END
