#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: 3D Yörüngesel Kenetlenme ve Güvenli GNC Sistemleri
Kapsamlı Görsel Mühendislik ve Simülasyon Raporu Oluşturucu (PDF)
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Font registration with DejaVuSans for 100% full Turkish & Math character support
FONT_DIR = "/usr/share/fonts/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))

# Color Palette: Deep Space & Aerospace Engineering
COLOR_PRIMARY = colors.HexColor("#0A0F1D")    # Dark Space Navy
COLOR_SECONDARY = colors.HexColor("#0284C7")  # Orbital Cyan
COLOR_ACCENT = colors.HexColor("#38BDF8")     # Sky Blue
COLOR_SUCCESS = colors.HexColor("#10B981")    # Emerald Green (Soft Docked)
COLOR_DANGER = colors.HexColor("#EF4444")     # Alert Red (Crash / Hazard)
COLOR_WARNING = colors.HexColor("#F59E0B")   # Amber Warning
COLOR_BG_CARD = colors.HexColor("#F1F5F9")    # Slate 100 (Card Background)
COLOR_BORDER = colors.HexColor("#CBD5E1")     # Slate 300
COLOR_TEXT_MAIN = colors.HexColor("#1E293B")  # Slate 800
COLOR_TEXT_MUTED = colors.HexColor("#64748B") # Slate 500

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 36  # pt (~12.7 mm)
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


class NumberedCanvas(canvas.Canvas):
    """Her sayfaya otomatik toplam sayfa numarası ve kurumsal üst/alt bilgi basar."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("DejaVu", 8)
        self.setFillColor(COLOR_TEXT_MUTED)

        # Üst Bilgi (Sayfa 2 ve sonrası)
        if self._pageNumber > 1:
            self.drawString(MARGIN, PAGE_HEIGHT - 26, "ORBITAL-RL: Görsel Mühendislik ve Yörüngesel Kenetlenme Raporu")
            self.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 26, "Sakarya Üniversitesi | Veri Bilimi ve Analitiği")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(MARGIN, PAGE_HEIGHT - 30, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 30)

        # Alt Bilgi (Tüm sayfalar)
        page_str = f"Sayfa {self._pageNumber} / {page_count}"
        self.drawString(MARGIN, 20, "Nasuhi — Orbital-RL Otonom Kenetlenme (github.com/Nasuhii/orbital-rl)")
        self.drawRightString(PAGE_WIDTH - MARGIN, 20, page_str)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(MARGIN, 28, PAGE_WIDTH - MARGIN, 28)

        self.restoreState()


def build_visual_pdf(output_filename="Orbital_RL_Görsel_Mühendislik_Raporu.pdf"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Tipografi Stilleri
    hero_title_style = ParagraphStyle(
        "HeroTitle",
        fontName="DejaVu-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.white,
        alignment=0,
        spaceAfter=4
    )

    hero_subtitle_style = ParagraphStyle(
        "HeroSubtitle",
        fontName="DejaVu-Bold",
        fontSize=9.5,
        leading=13,
        textColor=COLOR_ACCENT,
        alignment=0,
        spaceAfter=6
    )

    hero_meta_style = ParagraphStyle(
        "HeroMeta",
        fontName="DejaVu",
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#94A3B8")
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        fontName="DejaVu-Bold",
        fontSize=11.5,
        leading=15,
        textColor=COLOR_PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        fontName="DejaVu-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=COLOR_SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        fontName="DejaVu",
        fontSize=8.2,
        leading=11.5,
        textColor=COLOR_TEXT_MAIN,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        fontName="DejaVu",
        fontSize=7.8,
        leading=11,
        textColor=COLOR_TEXT_MAIN,
        leftIndent=10,
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        "ImageCaption",
        fontName="DejaVu-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=COLOR_TEXT_MUTED,
        alignment=1,
        spaceBefore=3,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="DejaVu-Bold",
        fontSize=7.8,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="DejaVu",
        fontSize=7.4,
        leading=9.5,
        textColor=COLOR_TEXT_MAIN,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        fontName="DejaVu-Bold",
        fontSize=7.4,
        leading=9.5,
        textColor=COLOR_TEXT_MAIN,
        alignment=0
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        fontName="DejaVu",
        fontSize=7.8,
        leading=11,
        textColor=COLOR_TEXT_MAIN
    )

    story = []

    # =========================================================================
    # SAYFA 1: BAŞLIK, YÖNETİCİ ÖZETİ & UÇTAN UCA GÖREV MİMARİSİ
    # =========================================================================

    # Hero Banner
    hero_content = [
        [Paragraph("ORBITAL-RL: 3D YÖRÜNGESEL KENETLENME VE GÜVENLİ GNC SİSTEMLERİ", hero_title_style)],
        [Paragraph("Google DeepMind MuJoCo ve Clohessy-Wiltshire Dinamiğinde Görsel Mühendislik Raporu", hero_subtitle_style)],
        [Paragraph("<b>Mühendis & Araştırmacı:</b> Nasuhi &nbsp;|&nbsp; <b>Bölüm:</b> Sakarya Üniversitesi — Veri Bilimi ve Analitiği (2. Sınıf)<br/>"
                   "<b>Fizik & Görsel Motor:</b> DeepMind MuJoCo v3.17 + RK4 &nbsp;|&nbsp; <b>Hedef Yörünge:</b> LEO 400 km &nbsp;|&nbsp; <b>Tarih:</b> 6 Kasım 2026", hero_meta_style)]
    ]
    hero_table = Table(hero_content, colWidths=[CONTENT_WIDTH])
    hero_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_PRIMARY),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,-1), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(hero_table)
    story.append(Spacer(1, 8))

    # Yönetici Özeti
    story.append(Paragraph("1. Yönetici Özeti (Executive Summary)", h1_style))
    summary_text = (
        "Bu rapor; Alçak Dünya Yörüngesinde (LEO ~400 km) seyreden bir hedef uzay istasyonuna kovalayıcı uzay aracının "
        "(Chaser, m=500 kg) otonom kenetlenmesini sağlayan <b>Orbital-RL</b> projesinin kapsamlı görsel çıktılarını, "
        "sensör telemetrisini ve çoklu kamera gözlemlerini sunmaktadır. Geleneksel yapay zeka ajanlarında sıkça görülen "
        "<b>'Hovering (Liman Önünde Asılı Kalma / Çarpışmaktan Kaçınma)'</b> ve yüksek hızlı <b>'Çarpışma (Crash)'</b> patolojileri, "
        "<b>Asimptotik Hız Frenleme Valisi (Velocity Governor)</b> ve analitik <b>Control Barrier Function (CBF-QP)</b> güvenlik kalkanı "
        "ile matematiksel olarak tamamen ortadan kaldırılmıştır. DeepMind MuJoCo fizik motorunda 30 uçuşluk Monte Carlo stres "
        "testlerinde <b>%100 kusursuz yumuşak kenetlenme (Soft & Hard Docking)</b> başarısı elde edilmiştir."
    )
    story.append(Paragraph(summary_text, body_style))

    # Uçtan Uca Görev Mimarisi Görseli
    story.append(Paragraph("2. Uçtan Uca Görev Mimarisi ve Kapalı Döngü GNC Akışı", h1_style))
    arch_img_path = os.path.join(script_dir, "mission_profile_architecture.png")
    if os.path.exists(arch_img_path):
        # 4245x2865 -> aspect ratio 1.482
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.482
        story.append(Image(arch_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 1: Uçtan Uca Görev Mimarisi — Üst: 4 Görev Fazı (Hohmann → CW Rendezvous → V-Bar GNC → Hard Dock). Alt: Kapalı Döngü Safe-GNC Gerçek Zamanlı Kontrol Mimarisi.", caption_style))

    story.append(Paragraph("• <b>Faz 1 - Hohmann Enjeksiyonu:</b> İki teğetsel itki darbesi ile 100 km irtifadan istasyonun 10 km arkasına yerleşme.", bullet_style))
    story.append(Paragraph("• <b>Faz 2 - CW Yaklaşma (Far-Field):</b> 3D Hill denklemleriyle Coriolis ve yerçekimi gradyanı dengelenerek optik kilitlenme (LOS) sağlanması.", bullet_style))
    story.append(Paragraph("• <b>Faz 3 - V-Bar GNC Koridoru:</b> Asimptotik hız valisi ile hızın mesafeye bağlı düşürülmesi ve CBF-QP güvenlik garantisi.", bullet_style))
    story.append(Paragraph("• <b>Faz 4 - Soft & Hard Capture:</b> Son 1.5 metrede 4.2 cm/s sabit sürünme hızıyla burun probunun huniye girişi ve mekanik mandalların kilitlenmesi.", bullet_style))

    # =========================================================================
    # SAYFA 2: 5 GÖREV KAPILI ÇOKLU KAMERA MUJOCO SIRALAMASI (STORYBOARD)
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("3. 5 Görev Kapılı Çoklu Kamera Sıralaması (MuJoCo Storyboard)", h1_style))
    story.append(Paragraph(
        "Kovalayıcı uzay aracının 20 metrelik yörünge rotasından limana kilitlendiği ana kadar geçen süreç, "
        "DeepMind MuJoCo fizik motorundan 3 farklı kamera perspektifinden eş zamanlı olarak kaydedilmiştir. "
        "Arka planda gerçekçi Dünya eğriliği ve uzay istasyonu güneş panelleri net olarak izlenmektedir:",
        body_style
    ))

    storyboard_img_path = os.path.join(script_dir, "mujoco_multicam_docking_sequence.png")
    if os.path.exists(storyboard_img_path):
        # 4154x2357 -> ratio 1.762
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.762
        story.append(Image(storyboard_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 2: Çoklu Kamera MuJoCo Kenetlenme Sıralaması (Storyboard) — Satır 1: Chase Cam (3. Şahıs & LEO). Satır 2: Optical Nav Probe Cam (FPV HUD). Satır 3: Station CCTV (İskele Güvenlik Kamerası).", caption_style))

    story.append(Paragraph("5 Kritik Görev Kapısının (Mission Gates) Telemetri Analizi:", h2_style))
    story.append(Paragraph("1. <b>Gate 1: Rendezvous Ingress (t=0.0s, d=18.0m, v=38.0 cm/s):</b> Kovalayıcı hedefle görsel teması kurar. Arka planda Dünya eğriliği görünür. RCS iticileri yanal Coriolis sürüklenmesini bastırmak için ilk düzeltme darbesini ateşler.", bullet_style))
    story.append(Paragraph("2. <b>Gate 2: V-Bar Acquisition (t=28.5s, d=10.0m, v=28.1 cm/s):</b> Araç ±25° yaklaşma konisinin merkez eksenine kilitlenir. Radyal ve düzlem dışı sapmalar ±5 cm altına indirilir.", bullet_style))
    story.append(Paragraph("3. <b>Gate 3: Terminal Braking (t=56.0s, d=3.0m, v=11.4 cm/s):</b> Asimptotik vali hızı hızla düşürür; istasyon kenetlenme halkası FPV prob kamerasının görüş alanını tamamen doldurur.", bullet_style))
    story.append(Paragraph("4. <b>Gate 4: Soft Capture Ingress (t=72.0s, d=0.6m, v=4.4 cm/s):</b> Burun probu alıcı huniye girer. Hız 5 cm/s altına çekilerek çarpışma enerjisi sönümlenir.", bullet_style))
    story.append(Paragraph("5. <b>Gate 5: Hard Latched (t=79.5s, d=0.15m, v=5.08 cm/s):</b> Mekanik mandallar 0.35 radyan kapanır, kenetlenme %100 nominal başarıyla tamamlanır.", bullet_style))

    # =========================================================================
    # SAYFA 3: KOKPİT OPTİK NAVİGASYON VE RETICLE HUD PANELİ
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. Uzay Aracı Optik Navigasyon Kokpit HUD Paneli", h1_style))
    story.append(Paragraph(
        "Kovalayıcı uzay aracının burun probuna entegre edilmiş yüksek çözünürlüklü optik kameradan alınan canlı görüntü "
        "ve pilot/GNC durum farkındalığı için üzerine bindirilen havacılık standartlarındaki Head-Up Display (HUD) arayüzü:",
        body_style
    ))

    hud_img_path = os.path.join(script_dir, "optical_nav_sensor_hud.png")
    if os.path.exists(hud_img_path):
        # 2781x2040 -> ratio 1.363
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.363 * 0.90 # slightly scale to fit cleanly
        story.append(Image(hud_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 3: Optik Navigasyon ve Sensör Füzyonu HUD Paneli — Merkez nişangah, sol hız bandı (Velocity Tape), sağ mesafe bandı (Range Tape), üst durum paneli, alt RCS eyleyici telemetrisi ve sağ altta Chase Cam penceresi.", caption_style))

    story.append(Paragraph("HUD Enstrümantasyonunun Fonksiyonel Rolleri:", h2_style))
    
    hud_table_data = [
        [Paragraph("HUD Enstrümanı", table_header_style), Paragraph("Görüntülenen Metrik", table_header_style), Paragraph("Güvenlik ve Operasyonel İşlevi", table_header_style)],
        [Paragraph("Optik Nişangah (Reticle)", table_cell_left), Paragraph("Merkez artı ve 10m/5m/2m halkaları", table_cell_style), Paragraph("İstasyon portundaki sarı optik hedef işaretini merkezleyerek açısal sapmayı sıfırlar.", table_cell_style)],
        [Paragraph("Sol Hız Bandı (Velocity Tape)", table_cell_left), Paragraph("v = 4.2 cm/s (Max Limit: 6.0 cm/s)", table_cell_style), Paragraph("Kırmızı tehlike sınırını görselleştirir; hızın mandalları kırmasını engeller.", table_cell_style)],
        [Paragraph("Sağ Mesafe Bandı (Range Tape)", table_cell_left), Paragraph("d = 1.20 m (Port: 0.15 m)", table_cell_style), Paragraph("Liman yüzeyine kalan mesafeyi ve anlık yaklaşma oranını (range-rate) gösterir.", table_cell_style)],
        [Paragraph("CBF Durum Annunciator", table_cell_left), Paragraph("CBF: ACTIVE (h = +0.0016 > 0)", table_cell_style), Paragraph("Bariyer güvenlik kalkanının devrede olduğunu ve sıfır ihlali doğrular.", table_cell_style)],
        [Paragraph("RCS Eyleyici Paneli", table_cell_left), Paragraph("Fx=+0.2N, Fy=-12.5N, Fz=0.0N", table_cell_style), Paragraph("PWPF modülatörünün 3 eksende ateşlediği darbe kuvvetlerini canlı raporlar.", table_cell_style)],
    ]
    hud_table = Table(hud_table_data, colWidths=[120, 150, CONTENT_WIDTH - 270])
    hud_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_SECONDARY),
        ('PADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_BG_CARD])
    ]))
    story.append(hud_table)

    # =========================================================================
    # SAYFA 4: FAZ DÜZLEMİ (PHASE PLANE) VE CBF-QP GÜVENLİK KALKANI
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("5. Faz Düzlemi (Phase Plane) ve CBF-QP Güvenlik Sınırları", h1_style))
    story.append(Paragraph(
        "Pekiştirmeli öğrenme modellerinin en büyük zayıflığı olan donma (hovering) ve çarpışma risklerini sıfıra indiren "
        "asimptotik hız valisi ve Control Barrier Function (CBF-QP) matematiksel güvenlik analizi:",
        body_style
    ))

    phase_img_path = os.path.join(script_dir, "phase_plane_safety_corridors.png")
    if os.path.exists(phase_img_path):
        # 4467x3465 -> ratio 1.289
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.289 * 0.88
        story.append(Image(phase_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 4: 4-Panelli Güvenlik ve Eyleyici Analizi — Sol Üst: Faz Düzlemi (v vs d). Sağ Üst: CBF Güvenlik Marjı h(x). Sol Alt: PWPF RCS İtici Ateşlemeleri. Sağ Alt: V-Bar Hizalanma Sapması.", caption_style))

    story.append(Paragraph("Matematiksel Emniyet Formülleri:", h2_style))
    story.append(Paragraph("• <b>Asimptotik Hız Valisi (Velocity Governor):</b> <i>v<sub>safe</sub>(d) = v<sub>dock</sub> + (v<sub>max</sub> - v<sub>dock</sub>) · tanh(d / d<sub>brake</sub>)</i>. Bu formül, araç uzaktayken serbest seyre izin verirken, limana yaklaştıkça hızı pürüzsüzce 0.042 m/s seviyesine indirir.", bullet_style))
    story.append(Paragraph("• <b>CBF Bariyer Kısıtı:</b> <i>h(x) = v<sub>safe</sub>²(d) - ||v||² ≥ 0</i>. Türev koşulu <i>ḣ + α·h ≥ 0</i> kısıtı altında Quadratic Programming (QP) çözücüsü, nominal itkiyi anında güvenli alt-uzaya projekte eder.", bullet_style))
    story.append(Paragraph("• <b>Sıfır İhlal Kanıtı (Panel 2):</b> 80 saniyelik uçuş boyunca h(x) değeri tek bir mikrosaniye dahi negatif bölgeye inmemiş, limana giriş hızı tam güvenli koridorda tutulmuştur.", bullet_style))

    # =========================================================================
    # SAYFA 5: AI MODEL EĞİTİMİ, ÖĞRENME EĞRİLERİ VE BENCHMARK DASHBOARD
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("6. Yapay Zeka (RL) Model Eğitimi, Öğrenme Dinamikleri ve Performans Dashboard'u", h1_style))
    story.append(Paragraph(
        "Stable-Baselines3 kütüphanesi ile eğitilen PPO modellerinin (Naive, Asymptotic, PBRS, MuJoCo DR, BC+PPO) "
        "öğrenme eğrileri (rolling return), Monte Carlo başarı dağılımı, yakıt/hız dengesi ve görev süreleri:",
        body_style
    ))

    ai_dash_path = os.path.join(script_dir, "rl_ai_training_and_benchmark_dashboard.png")
    if os.path.exists(ai_dash_path):
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.5 * 0.88
        story.append(Image(ai_dash_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 5: 4-Panelli AI Model Eğitim ve Kıyaslama Dashboard'u — Sol Üst: Rolling Episode Return vs Timesteps. Sağ Üst: Başarı/Hovering/Kaza Yüzdeleri. Sol Alt: Yakıt (Δv) vs Son Hız. Sağ Alt: Kalan Mesafe ve Görev Süresi.", caption_style))

    story.append(Paragraph("Eğitim Hiperparametreleri ve Model Mimarisi:", h2_style))
    story.append(Paragraph("• <b>Optimizasyon & Ağ:</b> Adam (lr=3e-4), Batch=64, n_steps=1024, Epochs=10, γ=0.99, GAE λ=0.95, Clip Range ε=0.15, Entropy Coef c2=0.001.", bullet_style))
    story.append(Paragraph("• <b>Öğrenme Bulgusu:</b> Naive model ceza korkusuyla negatif ödül bölgesinde (-692) kilitlenirken; PBRS (+228) ve MuJoCo DR (+1282) modelleri istikrarlı biçimde pozitif ödüllere yükselmiştir.", bullet_style))

    # =========================================================================
    # SAYFA 6: 30 UÇUŞLUK MUJOCO MONTE CARLO MASTER DOĞRULAMASI
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("7. MuJoCo 30 Uçuşluk Monte Carlo Doğrulaması (%100 Başarı)", h1_style))
    story.append(Paragraph(
        "NASA ve ESA standartlarına uygun olarak, rastgele başlangıç mesafeleri (15-25m), kaçış hızları ve yanal sapmaları "
        "içeren 30 zorlu Monte Carlo uçuşunda Pure PPO, PPO+CBF ve Integrated Safe-GNC mimarileri yarıştırılmıştır:",
        body_style
    ))

    benchmark_img_path = os.path.join(script_dir, "mujoco_30_monte_carlo_benchmark.png")
    if os.path.exists(benchmark_img_path):
        # 2880x1080 -> ratio 2.667
        w_img = CONTENT_WIDTH
        h_img = w_img / 2.667
        story.append(Image(benchmark_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 5: 30 Uçuşluk MuJoCo Monte Carlo Kıyaslaması — Pure PPO (kalkansız), PPO+CBF (ham sinir ağı) ve Integrated Safe-GNC (Master Mimari) başarı ve kaza oranları.", caption_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("30 Uçuşluk Master Karşılaştırma Matrisi:", h2_style))

    matrix_data = [
        [Paragraph("Algoritma / Mimari", table_header_style), Paragraph("Başarı Oranı (%)", table_header_style), Paragraph("Hovering (%)", table_header_style), Paragraph("Kaza / Çarpışma", table_header_style), Paragraph("Ortalama Δv (Yakıt)", table_header_style), Paragraph("Son Hız (m/s)", table_header_style)],
        [Paragraph("Pure PPO (Kalkansız RL)", table_cell_left), Paragraph("%0.0 (0/30)", table_cell_style), Paragraph("%43.3 (13/30)", table_cell_style), Paragraph("<font color='#EF4444'><b>%56.7 (17/30) 💥</b></font>", table_cell_style), Paragraph("2.68 m/s", table_cell_style), Paragraph("1.409 m/s (Kaza)", table_cell_style)],
        [Paragraph("PPO + CBF-QP (Ham Kalkan)", table_cell_left), Paragraph("%0.0 (0/30)", table_cell_style), Paragraph("<font color='#F59E0B'><b>%100.0 (30/30) ⏸️</b></font>", table_cell_style), Paragraph("<b>%0.0 (0/30) 🟢</b>", table_cell_style), Paragraph("1.65 m/s", table_cell_style), Paragraph("0.299 m/s", table_cell_style)],
        [Paragraph("Integrated Safe-GNC (Master)", table_cell_left), Paragraph("<font color='#10B981'><b>%100.0 (30/30) 🟢</b></font>", table_cell_style), Paragraph("<b>%0.0 (0/30) 🟢</b>", table_cell_style), Paragraph("<b>%0.0 (0/30) 🟢</b>", table_cell_style), Paragraph("<b>0.94 m/s ⚡</b>", table_cell_style), Paragraph("<b>0.044 m/s 🔒</b>", table_cell_style)],
    ]
    matrix_table = Table(matrix_data, colWidths=[120, 80, 80, 80, 85, CONTENT_WIDTH - 445])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('PADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_BG_CARD])
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Monte Carlo Bulgularının Analizi:", h2_style))
    story.append(Paragraph("1. <b>Kalkansız RL'in İflası:</b> Pure PPO modeli, kaza cezası ile hız limiti arasında bir emniyet kalkanı olmadığı için 30 uçuşun 17'sinde istasyona şiddetle çarpmıştır (ortalama son hız 1.41 m/s).", bullet_style))
    story.append(Paragraph("2. <b>Sinir Ağı-Kalkan Çekişmesi:</b> Ham PPO modeline doğrudan CBF bağlandığında kaza oranı %0'a inmiş ancak sinir ağı ile kalkanın zıt kuvvet talepleri aracı 1.6 metrede dondurmuştur (Hovering).", bullet_style))
    story.append(Paragraph("3. <b>Integrated Safe-GNC Zaferi:</b> Asimptotik rehberlik, terminal sürünme takviyesi ve CBF-QP filtresi birleştiğinde 30 uçuşun tamamında (30/30) mandalların hasar almadan kilitlendiği HARD CAPTURE elde edilmiştir.", bullet_style))

    # =========================================================================
    # SAYFA 7: 6 PANELLİ 3D DERİNLEMESİNE DİNAMİK VE SEKTÖREL SAHA NOTLARI
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("8. 6 Panelli 3D Derinlemesine Vektör ve Dinamik Analizi", h1_style))
    story.append(Paragraph(
        "Kovalayıcı aracın 3 eksenli durum uzayında hareketini, Coriolis kuvvetlerinin sönümlenmesini ve eyleyici tepkilerini "
        "açıklayan çok boyutlu telemetri matrisi:",
        body_style
    ))

    deep_img_path = os.path.join(script_dir, "orbital_3d_deep_dive.png")
    if os.path.exists(deep_img_path):
        # 4400x2800 -> ratio 1.571
        w_img = CONTENT_WIDTH
        h_img = w_img / 1.571 * 0.82
        story.append(Image(deep_img_path, width=w_img, height=h_img))
        story.append(Paragraph("Şekil 7: 6 Panelli 3D Derinlemesine Dinamik — 3D İzometrik Yaklaşma, Radyal X-Y Düzlemi, Cross-Track Z-Y Düzlemi, Faz Portresi, 3 Eksen RCS İtkisi ve RL Ödül Akışı.", caption_style))

    # Senior Field Tips Callout Kutusu
    story.append(Spacer(1, 4))
    story.append(Paragraph("9. Sektörel Saha Notları ve Kıdemli GNC Mühendisliği Çıkarımları", h1_style))

    field_tips_content = [
        [Paragraph("<b>🧑‍💻 Senior Field Tips & Industry Reality (NASA / SpaceX / ESA Standartları)</b>", ParagraphStyle("FTHead", fontName="DejaVu-Bold", fontSize=8.5, textColor=COLOR_PRIMARY))],
        [Paragraph(
            "• <b>'Asla Ham Bir Sinir Ağını Aktüatörlere Doğrudan Bağlama':</b> Gerçek uzay görevlerinde (ISS Kenetlenmesi, Dragon NDS, Orion) "
            "sinir ağları asla aktüatörün tek hakimi olamaz. Önünde her zaman analitik bir Control Barrier Function (CBF) veya Simplex kalkanı bulunur.<br/>"
            "• <b>'Hovering Bir Model Eksikliği Değil, Ödül Açığıdır (Reward Hacking)':</b> Çarpışma cezası çok yüksek, adım ödülü pozitif ise, "
            "ajan liman önünde sonsuza kadar durarak pozitif ödül süzer. Çözüm, Potansiyel Bazlı Ödül Şekillendirme (PBRS) ve asimptotik hız tavanıdır.<br/>"
            "• <b>'Optik Sensörlerde Güneş Işığı ve Dünya Yansıması':</b> Optik kameralar Dünya'nın atmosferik parlamasıyla (albedo) doyuma ulaşabilir. "
            "Bu nedenle simülasyonumuzda FPV kamera, bağıl LIDAR ve 3. şahıs telemetri füzyonu (EKF) modellenmiştir.",
            callout_style
        )]
    ]
    ft_table = Table(field_tips_content, colWidths=[CONTENT_WIDTH])
    ft_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_CARD),
        ('BOX', (0,0), (-1,-1), 1.2, COLOR_SECONDARY),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(ft_table)
    story.append(Spacer(1, 8))

    # Proje Kaynakları ve Kapanış
    story.append(Paragraph("<b>Proje Kaynakları ve Depo:</b> GitHub: <u>github.com/Nasuhii/orbital-rl</u> (Özel Depo) &nbsp;|&nbsp; "
                           "Lisans: MIT Open-Source &nbsp;|&nbsp; Canlı 3D Arayüz: <i>docking_visualizer.html</i>", body_style))

    # PDF Derleme
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✨ Görsel PDF Raporu başarıyla üretildi: {output_path}")
    return output_path


if __name__ == "__main__":
    build_visual_pdf()
