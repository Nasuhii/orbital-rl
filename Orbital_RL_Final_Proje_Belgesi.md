TARGET DECK: data

# 🛰️ Orbital-RL: Nihai Proje Kapanış Belgesi & Mühendislik Raporu

> **Proje Deposu:** [github.com/Nasuhii/orbital-rl](https://github.com/Nasuhii/orbital-rl) 🔒  
> **Geliştirici & GNC / RL Araştırmacısı:** Nasuhii  
> **Kapsam:** LEO (~400 km) Otonom Uzay Aracı Kenetlenme (Autonomous Spacecraft Docking) — Fizik, Safe-RL, CBF-QP, PWPF & Anti-Overfit v5  
> **Teknoloji Yığını:** Python 3.10+, Gymnasium, Stable-Baselines3 (PPO), PyTorch, Google DeepMind MuJoCo, MJPEG/HTML5 Canlı Telemetri  
> **Durum:** 🟢 **TAMAMLANDI (Final Release — `Champion-PPO v5` + `Integrated Safe-GNC`)**

> [!ABSTRACT] 🎯 Yönetici Özeti — Problem, Çözüm ve Nihai Sonuç
> **Orbital-RL**, Alçak Dünya Yörüngesindeki (LEO, $400\text{ km}$) bir hedef uzay istasyonuna yaklaşan $500\text{ kg}$ kütleli kovalayıcı uzay aracının **otonom, güvenli ve yakıt verimli** kenetlenmesini sağlayan uçtan uca bir **Pekiştirmeli Öğrenme (RL) & GNC** sistemidir.
>
> Proje boyunca klasik RL yöntemlerinin kilitlendiği **Hovering Patolojisi (%100 asılı kalma, %0 başarı)**, **kalkansız çarpışma (%56.7 crash)** ve **dar dağılım ezberlemesi (overfitting)** problemleri sırasıyla çözülmüş; **3D Clohessy-Wiltshire + DeepMind MuJoCo**, **Asimptotik Hız Valisi + PBRS**, **CBF-QP Güvenlik Kalkanı**, **PWPF İtici Modülatörü** ve **Anti-Overfit `Champion-PPO v5`** mimarisiyle **ID %100.0**, **OOD Ortalama %92.0 (`Pure-Raw`)** ve **Integrated Safe-GNC %100.0 (30/30 Monte Carlo)** başarısına ulaşılmıştır.

---

## 🗺️ 1. Uçtan Uca Sistem Mimarisi (Master Akış Diyagramı)

```mermaid
flowchart TD
    subgraph G1["1. Fizik ve Yörünge Simülasyon Katmanı"]
        A1["3D Clohessy-Wiltshire Denklemleri<br/>(LEO 400 km, n = 0.00113 rad/s, RK4 dt = 0.5s)"] -->|Gymnasium Ortamı| A2["MuJoCo 3D Rijit Gövde Fiziği<br/>(orbital_docking.xml, m = 500 kg, ±15 N RCS)"]
        A2 -->|Sızıntısız DR & Sensör Gürültüsü| A3["9D / 16-Özellikli Gözlem Uzayı<br/>(asinh + LayerNorm Sıfır-Körlük Mimari)"]
    end

    subgraph G2["2. Yapay Zeka & Anti-Overfit RL Eğitimi (Champion-PPO v5)"]
        B1["4 Rejimli Geniş Zarf + 360° Fly-Around<br/>(Nominal, Kaçış vy>0, Yandan/Arkadan, Şok)"] -->|15 Epoch BC Warmstart| B2["Ölçeklenmiş Ödül & AdamW PPO<br/>(G_t ∈ [-15, +35], Critic EV R² ≈ 0.87)"]
        B2 -->|OOD Early Stopping| B3["ppo_mujoco_champion_v5_best_ood.zip<br/>(ID: %100.0 | OOD Pure-Raw: %92.0)"]
    end

    subgraph G3["3. Güvenlik Kalkanı & Gerçekçi Eyleyici"]
        C1["Nominal Politika Komutu u_rl<br/>([-15 N, +15 N]^3)"] -->|h(x) >= 0 Kısıtı| C2["CBF-QP Güvenlik Kalkanı<br/>(6.0 cm/s Çarpışma Sınırı Koruması)"]
        C2 -->|Sürekli Güvenli Kuvvet| C3["PWPF Schmitt Darbe Modülatörü<br/>(Chattering Önleme + 15 cm Mandal Kilidi)"]
    end

    subgraph G4["4. Doğrulama, Görsel Çıktı ve Canlı Dijital İkiz"]
        D1["X-Ray OOD & 30-Uçuş Monte Carlo<br/>(diagnose_overfitting_xray.py)"] -->|4 Kamera & HUD| D2["Görsel Mühendislik Raporları & PDF<br/>(generate_visual_mission_suite.py)"]
        D2 -->|30 FPS MJPEG + Telemetri| D3["Canlı Görev Kontrol Paneli<br/>(mujoco_live_server.py :8055)"]
    end

    G1 -->|Durum & Ödül| G2
    G2 -->|Politika Çıktısı| G3
    G3 -->|Fiziksel İtki| G4
```

---

## ⚙️ 2. Temel Fizik, Gereksinimler ve Başarı Kriterleri (`orbital_config.py`)

Tüm simülasyon, eğitim ve doğrulama betikleri tek bir merkezî konfigürasyon dosyasından ([orbital_config.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/orbital_config.py)) beslenir:

| Parametre | Değer | Mühendislik Anlamı |
| :--- | :---: | :--- |
| 🌍 **Yörünge İrtifası (LEO)** | `400 km` ($r = 6,778.1\text{ km}$) | ISS irtifası, ortalama açısal hız $n \approx 0.00113\text{ rad/s}$ |
| 🛰️ **Kovalayıcı Araç Kütlesi ($m$)** | `500.0 kg` | DR senaryolarında gerçek `body_mass` ±%25–%35 değişir |
| 🔥 **RCS İtki Tavanı ($F_{\max}$)** | `±15.0 N` (3 Eksen) | Maksimum ivme $a_{\max} = 0.03\text{ m/s}^2$ (uzun fren mesafesi) |
| ⏱️ **Kontrol Adımı ($dt$)** | `0.5 s` (`5 × 0.1 s` MuJoCo) | `300` adım = `150 s` nominal / `450` adım = `225 s` OOD bütçesi |
| 🎯 **Kenetlenme Yarıçapı ($r_{\text{dock}}$)** | `≤ 0.14 m` (`~15 cm`) | Hedef port mandallarının yakalama toleransı |
| 🛡️ **Maks. Temas Hızı ($v_{\max,\text{dock}}$)** | `≤ 0.06 m/s` (`6.0 cm/s`) | Üzerindeki her temas **Crash (Yapısal Hasar)** sayılır |
| 📐 **Görüş Koridoru (LOS Cone)** | `±25.0°` | V-Bar yaklaşma ekseni hizalama kısıtı |

---

## 🧪 3. Evrim Süreci: Hovering Patolojisinden `Champion-PPO v5`'e

Proje 5 ana teknik nesil boyunca iteratif olarak olgunlaştırılmıştır:

1. **🔴 Nesil 1 — `Naive PPO` (Doğrusal Ceza):** Ajan $-100$ çarpışma cezasından kaçmak için limana $2\text{–}8\text{ m}$ kala hızını sıfırlayıp süreyi doldurdu (**%100 Hovering, %0 Başarı**).
2. **🔵 Nesil 2 & 3 — `Asymptotic` & `PBRS PPO`:** Asimptotik hız valisi $v_{\text{safe}}(d) = v_{\text{dock}} + (v_{\max}-v_{\text{dock}})\tanh(d/d_{\text{brake}})$ ve Ng et al. (1999) potansiyel farkı $F = \gamma \Phi(s') - \Phi(s)$ ile durmanın negatif ödül olduğu kanıtlandı; ancak son $40\text{ cm}$'de PWPF ölü bölgesi aşılamadı.
3. **🟠 Nesil 4 — `MuJoCo DR` & `BC+PPO` (`v3` / `v4`):** MuJoCo temas fiziği, 30–120 uçuşluk **Behavioral Cloning (BC)** sıcak başlangıcı ve **CBF-QP + PWPF** entegrasyonu ile dar ID zarfında **%100.0** başarıya ulaşıldı.
4. **🛠️ Kırılma Noktası — X-Ray Overfitting Teşhisi (`diagnose_overfitting_xray.py`):**
   - `v4` modelinin `Pure-Raw` son hızının $\sigma = \pm 0.000038\text{ m/s}$ gibi aşırı ezberlenmiş olduğu ve `S5: 360° Fly-Around` senaryosunda **%0.0** başarıda kaldığı tespit edildi.
   - **4 Kritik Bug Yakalandı ve Düzeltildi:** (1) Büyük ödül ölçeğiyle `vf_loss` patlaması ve Aktör gradyan donması, (2) MuJoCo `body_mass` / `body_inertia` + `mj_setConst` eksikliği, (3) `obs[6:9]` sensör gürültüsü veri sızıntısı, (4) Canlı `qvel` şok enjeksiyonu hatası.
5. **🏆 Nesil 5 — `Champion-PPO v5` (`train_mujoco_champion_v5.py`):**
   - Ödül ölçeği $G_t \in [-15, +35]$ bandına çekildi (**Critic EV $R^2$: `0.000` → `0.869`**).
   - `OrbitalObsFeaturesExtractorV5` (`asinh` + 16 özellik + `LayerNorm`) ile **gözlem körlüğü %53 → %0.0** yapıldı.
   - 360° Fly-Around, Z-simetri veri artırımı, AdamW ve **OOD Early Stopping** ile genelleme boşluğu **3× küçültüldü**.

---

## 📊 4. Doğrulanmış Nihai Performans & Benchmark Sonuçları

### A. 30 Uçuşluk Monte Carlo Kıyaslaması (`rl_benchmark_stats.json` — Nominal Görev)

| Model Mimarisi | Başarı (%) | Hover (%) | Crash (%) | Son Hız | Son Mesafe | Ort. $\Delta v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🔴 **Naive PPO** | `%0.0` | `%100.0` | `%0.0` | `0.835 m/s` | `2.17 m` | `4.06 m/s` |
| 🟣 **PBRS PPO** | `%0.0` | `%100.0` | `%0.0` | `0.145 m/s` | `0.48 m` | `1.73 m/s` |
| 🟠 **MuJoCo DR PPO (Kalkansız)** | `%0.0` | `%43.3` | `%56.7` 💥 | `1.409 m/s` | `2.68 m` | `2.68 m/s` |
| 🟢 **Integrated Safe-GNC (Master)** | **`%100.0` (30/30)** | **`%0.0`** | **`%0.0`** | **`0.0441 m/s`** 🔒 | **`0.1375 m`** 🎯 | **`0.94 m/s`** ⚡ |

### B. `Champion-PPO v5` Sızıntısız OOD Stres Testi (`overfitting_xray_report.json` — Senaryo Başına 25 Uçuş)

| Test Senaryosu (25 Uçuş) | `v5` `Pure-Raw` Başarı | `v5` `Pure-Raw` $\Delta v$ | `v5` `CBF+PWPF` Başarı | `v5` `CBF` Müdahale |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **S1: Nominal ID** (`18–25 m`) | **`%100.0`** | `0.577 m/s` ⚡ | **`%100.0`** | **`%0.0`** |
| 🛠️ **S2: Gerçek Kütle ±%35 + %40 Eksen Arızası** | **`%100.0`** | `0.577 m/s` | **`%100.0`** | **`%0.0`** |
| 📡 **S3: 5cm/1.5cm Gürültü + ±4cm Bias + Gecikme** | `%60.0` *(Crash %20)* | `1.955 m/s` | `%56.0` *(Crash %4)* | `%2.2` |
| 🚀 **S4: 30–50m + Kaçış Hızı (`vy > 0`)** | **`%100.0`** | `1.203 m/s` | **`%96.0`** | `%0.4` |
| 🔄 **S5: 360° Fly-Around (`+Y` Dışı — Yandan/Arkadan)** | **`%100.0`** | `1.365 m/s` | **`%88.0`** | `%15.8` |
| ⚡ **S6: Gerçek `qvel` Çift Hız Şoku (`t=25s, 55s`)** | **`%100.0`** | `0.889 m/s` | **`%100.0`** | `%1.2` |
| 📊 **OOD ORTALAMA & GENELLEME BOŞLUĞU** | **`%92.0` (`Gap: %8.0`)** | — | **`%88.0` (`Gap: %12.0`)** | — |

> [!NOTE] 📌 Kalan Tek Bilinen Sınır (S3 Yüksek Gürültü)
> `Champion-PPO v5`, S1, S2, S4, S5 ve S6'da **`Pure-Raw` %100.0** başarıya ulaşmıştır. Tek zayıf nokta olan **S3 (5 cm konum / 1.5 cm/s hız gürültüsü + ±4 cm bias + 0.5 s gecikme)** senaryosunda başarı `%60.0`'dır. Bunun mimari sebebi mevcut politikanın tek adımlı (Markovian) gözlem kullanmasıdır; gelecek sürüm (v6) için **Frame Stacking (son 4–8 adım)** veya **LSTM / Kalman Filtresi** önerilmektedir.

---

## 📁 5. Teslimat Envanteri ve Çalıştırma Rehberi

### 📦 Ana Dosyalar ve Raporlar
- **Çekirdek Ortam & Güvenlik:** [orbital_config.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/orbital_config.py), [orbital_env.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/orbital_env.py), [orbital_mujoco_env.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/orbital_mujoco_env.py), [cbf_qp_shield.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/cbf_qp_shield.py), [pwpf_modulator.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/pwpf_modulator.py)
- **Eğitim & Diagnostik:** [train_mujoco_champion_v5.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/train_mujoco_champion_v5.py), [finetune_v5_terminal_braking.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/finetune_v5_terminal_braking.py), [diagnose_overfitting_xray.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/diagnose_overfitting_xray.py)
- **Nihai Model Checkpoint:** [ppo_mujoco_champion_v5_best_ood.zip](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/models/ppo_mujoco_champion_v5_best_ood.zip)
- **Canlı Dijital İkiz & Görselleştirme:** [mujoco_live_server.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/mujoco_live_server.py), [mujoco_mission_control.html](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/mujoco_mission_control.html), [generate_visual_mission_suite.py](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/generate_visual_mission_suite.py)
- **Alt Dokümantasyonlar & PDF'ler:**
  - [Orbital_RL_Proje_Ilerleme_ve_Sunum_Dosyasi.md](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/Orbital_RL_Proje_Ilerleme_ve_Sunum_Dosyasi.md)
  - [RL_Model_Egitimi_ve_Altyapi_Raporu.md](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/RL_Model_Egitimi_ve_Altyapi_Raporu.md)
  - [Orbital_RL_Overfitting_ve_Metrik_Analizi.md](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/Orbital_RL_Overfitting_ve_Metrik_Analizi.md)
  - [Orbital_RL_Görsel_Mühendislik_Raporu.pdf](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/Orbital_RL_Görsel_Mühendislik_Raporu.pdf) & [Orbital_RL_Overfitting_ve_Metrik_Analizi.pdf](file:///mnt/E2E0836DE083472F/Users/Nasuhi/Belgelerim/Obsidian%20Vault/Areas/Software/orbital-rl/Orbital_RL_Overfitting_ve_Metrik_Analizi.pdf)

### 🚀 Hızlı Çalıştırma Komutları
```bash
# 1. Canlı 30 FPS MuJoCo Görev Kontrol Arayüzünü Başlatma (http://localhost:8055/)
python mujoco_live_server.py 8055

# 2. X-Ray OOD Stres Testini Yeniden Koşturma
python diagnose_overfitting_xray.py

# 3. Görsel Görev Paketini (PNG/MP4/GIF) ve Mühendislik PDF'ini Üretme
python generate_visual_mission_suite.py && python generate_visual_pdf.py
```

---

> [!EXAMPLE] 🧑‍💻 Senior Field Tips & Industry Reality
> - **Acemi Hatası:** Yalnızca dar eğitim zarfında (`ID`) %100 başarı alıp projeyi "bitti" saymak; sensör gürültüsü sızıntılarını veya sabit kütle varsayımlarını kontrol etmemek.
> - **Senior Mühendis Yaklaşımı:** Uzay/havacılık GNC sistemlerinde sinir ağı tek başına uçuş bilgisayarına bağlanmaz (**Safety-Filtered Hybrid GNC**). Politika hem **kalkansız (`Pure-Raw`)** hem **kalkanlı (`CBF+PWPF`)** modda, **Held-Out OOD** senaryolarında (360° yaklaşma, kütle/itki arızası, hız şokları) doğrulanır ve teslim belgesi hem teoriyi hem sınırlamaları (S3 gürültü sınırı gibi) şeffaf biçimde raporlar.

---

## 📖 B2+ İngilizce Terminoloji ve Telaffuz

| İngilizce Terim | Türkçe Anlamı | IPA Telaffuz | Projedeki Bağlamı |
| :--- | :--- | :--- | :--- |
| **Autonomous Rendezvous** | Otonom Buluşma / Yanaşma | `/ɔːˈtɒn.ə.məs ˈrɒn.deɪ.vuː/` | Kovalayıcı aracın hedef istasyona kendi kendine yaklaşması |
| **Reward Shaping** | Ödül Şekillendirme | `/rɪˈwɔːd ˈʃeɪ.pɪŋ/` | Ajanı hovering'den kurtarmak için asimptotik ve PBRS ödül tasarımı |
| **Domain Randomization** | Alan Rastgeleleştirme | `/dəˈmeɪn ˌræn.də.maɪˈzeɪ.ʃən/` | Sim-to-real genelleme için kütle, itki ve gürültünün rastgele değiştirilmesi |
| **Out-of-Distribution (OOD)** | Dağılım Dışı | `/ˌaʊt əv ˌdɪs.trɪˈbjuː.ʃən/` | Eğitimde hiç görülmemiş 360° açılar veya kaçış hızı senaryoları |

---

## 🃏 Anki Kartları

START
Temel
TARGET DECK: data
Ön: Orbital-RL projesinde `Champion-PPO v5` modelinin `v4`'e göre 360° Fly-Around ve OOD genelleme başarısını %0'dan %92-100 bandına çıkaran 3 temel mühendislik düzeltmesi nedir?
Geri: 1. <b>Ödül Ölçekleme ($G_t \in [-15, +35]$):</b> `vf_loss` patlamasını ve Aktör gradyan donmasını çözdü (Critic $R^2: 0.000 \to 0.869$).<br>2. <b>Sıfır-Körlük Gözlem (`asinh` + LayerNorm):</b> `|x|, |z| > 2.5 m` doygunluk körlüğünü %53'ten %0.0'a indirdi.<br>3. <b>Sızıntısız DR + OOD Early Stopping:</b> Gerçek `mj_setConst` kütle DR'si, gürültülü `obs[6:9]` ve 360° fly-around eğitimi uygulandı.
Tags: from_obsidian, data, rl, orbital_rl, champion_ppo_v5
END

---

START
Temel
TARGET DECK: ing kelimeler
Ön: Out-of-Distribution (OOD)
Geri: Dağılım Dışı (Eğitim verisinde yer almayan yeni senaryo/koşul)<br>IPA: /ˌaʊt əv ˌdɪs.trɪˈbjuː.ʃən/
Tags: from_obsidian, vocab
END
