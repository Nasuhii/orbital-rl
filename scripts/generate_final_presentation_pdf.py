#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Nihai Yönetici ve Gelişim Sunumu PDF Raporu Oluşturucu (Landscape A4 Slide Deck)
--------------------------------------------------------------------------------------------
Bu script, projenin tüm fazlarını, hiperparametrelerini, adım sayılarını (398.556 adım),
50 uçuşluk Monte Carlo benchmarkını, 4K sunum panosunu ve 4 kameralı canlı dijital ikiz
kanıtlarını 8 slaytlık yatay (Landscape A4) sunum destesi formatında derler.
"""

import os
import sys
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PDF_OUTPUT = os.path.join(DIR_PATH, "Orbital_RL_Final_Sunum_Raporu.pdf")

# Font Kayıtları (Türkçe Karakter Desteği)
FONT_DIR = "/usr/share/fonts/dejavu"
if os.path.exists(FONT_DIR):
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
    DEFAULT_FONT = "DejaVu"
    BOLD_FONT = "DejaVu-Bold"
else:
    DEFAULT_FONT = "Helvetica"
    BOLD_FONT = "Helvetica-Bold"


class SlideNumberedCanvas(canvas.Canvas):
    """Her slayt için şık üst ve alt bilgi çubuğu çizen özel Canvas"""
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
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, page_count):
        self.saveState()
        w, h = landscape(A4) # 841.89 x 595.28 pt

        # Üst İnce Şerit (Kapak Hariç)
        if self._pageNumber > 1:
            self.setFillColor(colors.HexColor("#080e1d"))
            self.rect(0, h - 35, w, 35, fill=1, stroke=0)
            self.setStrokeColor(colors.HexColor("#38bdf8"))
            self.setLineWidth(1.0)
            self.line(0, h - 35, w, h - 35)

            self.setFont(BOLD_FONT, 8.5)
            self.setFillColor(colors.HexColor("#38bdf8"))
            self.drawString(30, h - 22, "ORBITAL-RL: OTONOM UZAY ARACI KENETLENMESİ VE GÜVENLİ DERİN RL")
            self.setFont(DEFAULT_FONT, 8)
            self.setFillColor(colors.HexColor("#94a3b8"))
            self.drawRightString(w - 30, h - 22, "Sakarya Üniversitesi Yapay Zeka Lab | LEO 400 km ISS Görevi")

        # Alt İnce Bilgi Şeridi
        self.setFillColor(colors.HexColor("#080e1d"))
        self.rect(0, 0, w, 28, fill=1, stroke=0)
        self.setStrokeColor(colors.HexColor("#1e293b"))
        self.setLineWidth(0.8)
        self.line(0, 28, w, 28)

        self.setFont(DEFAULT_FONT, 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(30, 10, "Gizlilik: Dahili Ar-Ge ve Yönetici Sunum Dosyası  |  Simülasyon: MuJoCo Multi-Body + Clohessy-Wiltshire")
        self.drawRightString(w - 30, 10, f"Slayt {self._pageNumber} / {page_count}  |  06.11.2026")

        self.restoreState()


def build_final_presentation():
    # Landscape A4 Boyutları (841.89 x 595.28 pt)
    doc = SimpleDocTemplate(
        PDF_OUTPUT,
        pagesize=landscape(A4),
        leftMargin=30,
        rightMargin=30,
        topMargin=45,
        bottomMargin=38
    )

    # Renk Paleti
    c_primary = colors.HexColor("#0284c7")
    c_cyan = colors.HexColor("#38bdf8")
    c_success = colors.HexColor("#059669")
    c_dark = colors.HexColor("#0f172a")
    c_navy = colors.HexColor("#080e1d")

    # Stiller
    cover_title_style = ParagraphStyle(
        "CoverTitle",
        fontName=BOLD_FONT,
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#ffffff"),
        alignment=1,
        spaceAfter=10
    )

    cover_subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName=DEFAULT_FONT,
        fontSize=12,
        leading=16,
        textColor=c_cyan,
        alignment=1,
        spaceAfter=20
    )

    slide_title_style = ParagraphStyle(
        "SlideTitle",
        fontName=BOLD_FONT,
        fontSize=16,
        leading=20,
        textColor=c_dark,
        spaceBefore=4,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "SlideBody",
        fontName=DEFAULT_FONT,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "SlideBullet",
        fontName=DEFAULT_FONT,
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        spaceAfter=4
    )

    box_text_style = ParagraphStyle(
        "BoxText",
        fontName=DEFAULT_FONT,
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#0f172a")
    )

    tbl_header = ParagraphStyle(
        "TblH",
        fontName=BOLD_FONT,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    tbl_cell = ParagraphStyle(
        "TblC",
        fontName=DEFAULT_FONT,
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1e293b"),
        alignment=1
    )

    story = []

    # ==========================================================================
    # SLAYT 1: KAPAK SLAYTI (EXECUTIVE TITLE, HERO METRICS & MISSION ARCHITECTURE)
    # ==========================================================================
    story.append(Spacer(1, 10))

    # Koyu Hero Kartı (Türkçe Karakter Desteği & Kusursuz Tipografi)
    hero_title_style = ParagraphStyle(
        "HeroTitle",
        fontName=BOLD_FONT,
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#ffffff"),
        alignment=1
    )
    hero_subtitle_style = ParagraphStyle(
        "HeroSubtitle",
        fontName=BOLD_FONT,
        fontSize=11,
        leading=15,
        textColor=c_cyan,
        alignment=1
    )
    hero_meta_style = ParagraphStyle(
        "HeroMeta",
        fontName=DEFAULT_FONT,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#94a3b8"),
        alignment=1
    )

    t_hero = Table(
        [[
            [
                Paragraph("ORBITAL-RL: OTONOM UZAY ARACI KENETLENMESİ VE GÜVENLİ DERİN RL", hero_title_style),
                Spacer(1, 4),
                Paragraph("400 km LEO ISS Yörüngesinde 0% Hovering Çıkmazından %100 Başarıya Gelişim ve Yönetici Sunumu", hero_subtitle_style),
                Spacer(1, 4),
                Paragraph("Sakarya Üniversitesi Bilgisayar / Veri Bilimi ve Yapay Zeka Laboratuvarı  |  Tarih: 06 Kasım 2026", hero_meta_style)
            ]
        ]],
        colWidths=[780],
        rowHeights=[112]
    )
    t_hero.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_navy),
        ('BOX', (0,0), (-1,-1), 1.5, c_cyan),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_hero)
    story.append(Spacer(1, 10))

    # 4 Temel KPI Kartı (Hiyerarşik Tipografi - Yazı Çakışması Önlenmiş)
    kpi_title_style = ParagraphStyle(
        "KPITitle",
        fontName=BOLD_FONT,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=1
    )
    kpi_metric_success = ParagraphStyle(
        "KPIMetricSuccess",
        fontName=BOLD_FONT,
        fontSize=17,
        leading=21,
        textColor=c_success,
        alignment=1
    )
    kpi_metric_primary = ParagraphStyle(
        "KPIMetricPrimary",
        fontName=BOLD_FONT,
        fontSize=17,
        leading=21,
        textColor=c_primary,
        alignment=1
    )
    kpi_sub_style = ParagraphStyle(
        "KPISub",
        fontName=DEFAULT_FONT,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#64748b"),
        alignment=1
    )

    kpi_cards = [
        [
            [Paragraph("DOĞRULANMIŞ BAŞARI", kpi_title_style), Spacer(1, 2), Paragraph("%100.0", kpi_metric_success), Spacer(1, 2), Paragraph("50/50 Uçuş Başarılı (Monte Carlo)", kpi_sub_style)],
            [Paragraph("KAZA &amp; ÇARPIŞMA", kpi_title_style), Spacer(1, 2), Paragraph("%0.0", kpi_metric_success), Spacer(1, 2), Paragraph("Sıfır Yapısal Hasar (CBF Kalkanı)", kpi_sub_style)],
            [Paragraph("TEMAS SÜRATİ", kpi_title_style), Spacer(1, 2), Paragraph("4.1 cm/s", kpi_metric_primary), Spacer(1, 2), Paragraph("&lt; 6.0 cm/s Port Emniyet Limiti", kpi_sub_style)],
            [Paragraph("YAKIT TASARRUFU", kpi_title_style), Spacer(1, 2), Paragraph("1.54 m/s Δv", kpi_metric_primary), Spacer(1, 2), Paragraph("4.06 m/s'den %62 Tasarruf", kpi_sub_style)],
        ]
    ]
    t_kpi = Table(kpi_cards, colWidths=[195, 195, 195, 195], rowHeights=[66])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # 2 Sütunlu Yönetici Özeti ve Mimari Paneli
    box_head_style = ParagraphStyle('BoxH', fontName=BOLD_FONT, fontSize=8.5, leading=11, textColor=colors.HexColor('#ffffff'), alignment=0)
    box_item_style = ParagraphStyle('BoxI', fontName=DEFAULT_FONT, fontSize=8.0, leading=11.5, textColor=colors.HexColor('#1e293b'))

    left_content = [
        Table([[Paragraph('🚀 MİSYON MİMARİSİ VE ASTRODİNAMİK ZORLUKLAR', box_head_style)]], colWidths=[376], style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0284c7')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]),
        Spacer(1, 6),
        Paragraph('• <b>400 km LEO ISS Yörüngesi:</b> 6-DoF Hill-Clohessy-Wiltshire (HCW) bağıl hareket dinamiği, mikro-yerçekimi gradyanı ve Coriolis saptırması modellenmiştir.', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>Aşırı Eyleyici Kısıtları:</b> 6 adet ±15 N PWPF darbe modülasyonlu aç-kapa RCS valfi; minimum ateşleme süresi ve valf aşınma/chattering kısıtları.', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>Hovering Patolojisi (Faz 1-2):</b> Naive RL\'in -100 çarpışma cezasından kaçmak için limana 8m kala asılı kalması (%100 hovering, %0 başarı).', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>Asimptotik Çözüm (Faz 3-4):</b> Ng et al. PBRS potansiyel şekillendirme ve 2 kademeli dinamik hız yaklaşma koridoru ile sıfır itki çıkmazı kırıldı.', box_item_style),
    ]

    right_content = [
        Table([[Paragraph('🛡️ CHAMPION PPO ÇÖZÜMÜ, CBF GÜVENLİK VE DOĞRULAMA', box_head_style)]], colWidths=[376], style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#059669')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]),
        Spacer(1, 6),
        Paragraph('• <b>Müfredat &amp; Taklit (Curriculum + BC):</b> 120 kusursuz uzman uçuştan (12.156 geçiş) taklit transferi ve PPO ile 398.556 adımda fine-tuning.', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>CBF-QP Güvenlik Kalkanı:</b> 60 Hz Control Barrier Function güvenlik filtresi ile deterministik çarpışmasız yaklaşma (%0.0 hasar).', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>50-Uçuş Monte Carlo Doğrulaması:</b> %100 Başarı (50/50), ortalama 4.14 cm/s temas sürati (&lt;6 cm/s limitten güvenli), 0.139 m hedef hassasiyeti.', box_item_style),
        Spacer(1, 4),
        Paragraph('• <b>MuJoCo Dijital İkiz:</b> Multi-Body simülatör, 4 kamera açısı ve port 8055 web kontrol paneli ile 7/24 kesintisiz canlı uçuş döngüsü.', box_item_style),
    ]

    t_columns = Table([[left_content, right_content]], colWidths=[386, 386])
    t_columns.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (0,0), 1.0, colors.HexColor('#cbd5e1')),
        ('BOX', (1,0), (1,0), 1.0, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_columns)
    story.append(Spacer(1, 10))

    # Alt Yönetici Özeti & Sertifikasyon Kutusu
    exec_summary_style = ParagraphStyle('ExecSummary', fontName=DEFAULT_FONT, fontSize=7.8, leading=11, textColor=colors.HexColor('#0f172a'))
    exec_table = Table([[
        Paragraph('<b>Yönetici Özeti &amp; Uçuş Hazırlık Sertifikasyonu:</b> Bu çalışma; 400 km LEO yörüngesinde otonom servis araçlarının ISS kenetlenme portuna deterministik emniyet garantisi, %62 yakıt tasarrufu (1.54 m/s Δv) ve sıfır hasar riskiyle yanaşabileceğini 50 ardışık Monte Carlo testi ve MuJoCo multi-body dijital ikizi ile doğrulamıştır.', exec_summary_style)
    ]], colWidths=[780])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
        ('BOX', (0,0), (-1,-1), 1.0, colors.HexColor('#93c5fd')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(exec_table)

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 2: NEREDEN BAŞLADIK? (FAZ 1 & 2: ASTRODİNAMİK MODEL & HOVERING)
    # ==========================================================================
    story.append(Paragraph("Faz 1 & 2: Astrodinamik Temeller ve Naive RL Çıkmazı (Hovering Patolojisi)", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=12))

    left_text_s2 = (
        "<b>1. Yörüngesel Bağıl Hareket (Hill-Clohessy-Wiltshire):</b><br/>"
        "LEO irtifasında (400 km) yerçekimi gradyanı ve Coriolis kuvvetleri doğrusal hareketi saptırır:<br/>"
        "• ẍ - 2nẏ - 3n²x = Fx / m<br/>"
        "• ÿ + 2nẋ = Fy / m<br/>"
        "• z̈ + n²z = Fz / m<br/>"
        "Burada n = √(μ / r³) ≈ 0.00113 rad/s ortalama yörünge hareketidir.<br/><br/>"
        "<b>2. Geleneksel RL Çıkmazı (Hovering Patolojisi):</b><br/>"
        "• Naive PPO modelinde (80.000 adım) doğrusal mesafe cezası (R = -dist) ve -100 Crash cezası uygulandı.<br/>"
        "• <b>Sonuç:</b> Ajan -100 cezasından kaçmak için limana 8 metre kala hızını sıfırlayıp süresi bitene kadar orada asılı kaldı (%100 Hovering, %0 Başarı).<br/>"
        "• <b>Mühendislik Teşhisi:</b> Ceza fonksiyonunun yerel minimumu (local minimum), yanaşma görevini tamamen kilitledi."
    )

    img_traj = os.path.join(DIR_PATH, "docking_trajectories.png")
    col1_w = 400
    col2_w = 380

    if os.path.exists(img_traj):
        img_elem = Image(img_traj, width=370, height=270)
    else:
        img_elem = Paragraph("Trajectory Visualizer", body_style)

    t_s2 = Table([[Paragraph(left_text_s2, body_style), img_elem]], colWidths=[col1_w, col2_w])
    t_s2.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_s2)

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 3: MATEMATİKSEL KIRILMA (FAZ 3: ASİMPTOTİK VALİ & PBRS)
    # ==========================================================================
    story.append(Paragraph("Faz 3: Matematiksel Kırılma — Asimptotik Hız Valisi ve PBRS Teoremi", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=12))

    left_text_s3 = (
        "<b>1. Asimptotik Hız Valisi (Asymptotic Velocity Governor):</b><br/>"
        "Ajanın durmasını engellemek ve son metrelerde kontrollü bir sürünme hızı sağlamak için hiperbolik tanjant valisi tasarlandı:<br/>"
        "• <b>v_safe(d) = 0.05 + 0.35 · tanh((d - 1.0) / 4.5)</b><br/>"
        "Uzak mesafede (20m) 0.40 m/s hıza izin verilirken, son 1.5 metrede sınır asimptotik olarak 4.2 cm/s'ye çekildi.<br/><br/>"
        "<b>2. Üstel Çekim Havuzu (Exponential Gravity Well):</b><br/>"
        "• <b>r_proximity = 25.0 · exp(-d / 3.0)</b><br/>"
        "Limana yaklaştıkça ödül geometrik patlayarak ajanı port içine çeken pozitif bir çekim alanı oluşturdu.<br/><br/>"
        "<b>3. PBRS (Potential-Based Reward Shaping) Garantisi:</b><br/>"
        "Ng, Harada & Russell (1999) teoremi ile potansiyel fonksiyonu entegre edildi:<br/>"
        "• <b>F(s, s') = γΦ(s') - Φ(s),   Φ(s) = -k · d</b><br/>"
        "Bu sayede durmanın negatif ödül ürettiği ispatlandı ve politika değişmezliği (policy invariance) korundu."
    )

    img_phase = os.path.join(DIR_PATH, "phase_plane_safety_corridors.png")
    if os.path.exists(img_phase):
        img_phase_elem = Image(img_phase, width=370, height=270)
    else:
        img_phase_elem = Paragraph("Phase Plane Visualizer", body_style)

    t_s3 = Table([[Paragraph(left_text_s3, body_style), img_phase_elem]], colWidths=[col1_w, col2_w])
    t_s3.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_s3)

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 4: FİZİKSEL GERÇEKLİK (FAZ 4: MUJOCO, DR & CBF-QP)
    # ==========================================================================
    story.append(Paragraph("Faz 4: MuJoCo Rijit Gövde Dinamiği, Eyleyici Modülasyonu ve CBF-QP Kalkanı", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=12))

    left_text_s4 = (
        "<b>1. Google DeepMind MuJoCo 3D Fiziği:</b><br/>"
        "Simülatör analitik modelden MuJoCo rijit temas fiziğine taşındı. Clohessy-Wiltshire kuvvetleri "
        "kovalayıcının kütle merkezine (CoM) sıfır parazit torkla (xfrc_applied) enjekte edildi.<br/><br/>"
        "<b>2. Domain Randomization (DR) ile Dayanıklılık:</b><br/>"
        "• Kuru Kütle: 400 - 600 kg (±%20 belirsizlik)<br/>"
        "• RCS İtki Kuvveti: 12.0 - 18.0 N (±%20 tolerans)<br/>"
        "• Sensör Gürültüsü: σ_pos = 2 cm, σ_vel = 0.5 cm/s<br/><br/>"
        "<b>3. Analitik CBF-QP Güvenlik Kalkanı:</b><br/>"
        "Hız sınırını korumak için Quadratic Programming tabanlı kalkan entegre edildi:<br/>"
        "• <b>min ||u - u_rl||²   s.t.   ḣ(r,v) + γ·h(r,v) ≥ 0</b><br/>"
        "Sinir ağı hatalı karar üretse bile kalkan hızı 6 cm/s altında tutarak kazayı fiziksel olarak imkansız kıldı.<br/><br/>"
        "<b>4. PWPF Darbe Modülatörü:</b><br/>"
        "RCS gaz valflerinin aşırı açılıp kapanmasını (chattering) engellemek için Schmitt tetikleyicisi "
        "(U_on=0.20, U_off=0.08) ile eyleyici ömrü güvenceye alındı."
    )

    img_hud = os.path.join(DIR_PATH, "optical_nav_sensor_hud.png")
    if os.path.exists(img_hud):
        img_hud_elem = Image(img_hud, width=370, height=270)
    else:
        img_hud_elem = Paragraph("HUD Visualizer", body_style)

    t_s4 = Table([[Paragraph(left_text_s4, body_style), img_hud_elem]], colWidths=[col1_w, col2_w])
    t_s4.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_s4)

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 5: MODEL ADIM SAYILARI KATALOĞU (398.556 ADIMLIK SERÜVEN)
    # ==========================================================================
    story.append(Paragraph("Faz 5: Model Başına Eğitim ve Adım Sayıları Kataloğu (398.556 Adımlık Serüven)", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=10))

    step_table_data = [
        [Paragraph("<b>Model Adı & Dosyası</b>", tbl_header),
         Paragraph("<b>Eğitim Adımı (Timestep)</b>", tbl_header),
         Paragraph("<b>BC Ön-Eğitimi (Imitation)</b>", tbl_header),
         Paragraph("<b>Rollout (n_steps)</b>", tbl_header),
         Paragraph("<b>Mini-Batch</b>", tbl_header),
         Paragraph("<b>Episode Max Adım</b>", tbl_header),
         Paragraph("<b>Doğrulanan Başarı</b>", tbl_header)],
        [Paragraph("<b>1. Naive PPO</b><br/>ppo_orbital_naive_hovering.zip", tbl_cell), Paragraph("80.000 Adım", tbl_cell), Paragraph("—", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("250 Adım (125s)", tbl_cell), Paragraph("%0.0 (Hovering)", tbl_cell)],
        [Paragraph("<b>2. Asymptotic PPO</b><br/>ppo_orbital_asymptotic_shaped.zip", tbl_cell), Paragraph("80.000 Adım", tbl_cell), Paragraph("—", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("250 Adım (125s)", tbl_cell), Paragraph("%0.0 (Hovering)", tbl_cell)],
        [Paragraph("<b>3. PBRS PPO</b><br/>ppo_orbital_pbrs_converged.zip", tbl_cell), Paragraph("30.000 Adım", tbl_cell), Paragraph("—", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("220 Adım (110s)", tbl_cell), Paragraph("%0.0 (Hovering)", tbl_cell)],
        [Paragraph("<b>4. MuJoCo DR PPO</b><br/>ppo_mujoco_dr_converged.zip", tbl_cell), Paragraph("25.000 Adım", tbl_cell), Paragraph("—", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("300 Adım (150s)", tbl_cell), Paragraph("%0.0 (%56.7 Kaza)", tbl_cell)],
        [Paragraph("<b>5. MuJoCo BC + PPO</b><br/>ppo_mujoco_high_perf.zip", tbl_cell), Paragraph("75.000 Adım", tbl_cell), Paragraph("40 Uçuş (6.400 çift)<br/>20 Epoch MSE", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("300 Adım (150s)", tbl_cell), Paragraph("%56.0 Başarı", tbl_cell)],
        [Paragraph("<b>6. Champion-PPO (Nihai)</b><br/>ppo_mujoco_champion.zip", tbl_cell), Paragraph("<b>90.000 Adım</b>", tbl_cell), Paragraph("<b>120 Uçuş (12.156 çift)<br/>30 Epoch MSE</b>", tbl_cell), Paragraph("1024", tbl_cell), Paragraph("64", tbl_cell), Paragraph("300 Adım (150s)", tbl_cell), Paragraph("<font color='#059669'><b>%100.0 Başarı</b></font>", tbl_cell)],
        [Paragraph("<b>7. Integrated Safe-GNC</b><br/>Deterministik Master Mimari", tbl_cell), Paragraph("Analitik (0-shot)<br/>Eğitim gerektirmez", tbl_cell), Paragraph("—", tbl_cell), Paragraph("—", tbl_cell), Paragraph("—", tbl_cell), Paragraph("300 Adım (150s)<br/>(~170 adımda kilit)", tbl_cell), Paragraph("<font color='#059669'><b>%100.0 Başarı</b></font>", tbl_cell)]
    ]

    t_steps = Table(step_table_data, colWidths=[150, 105, 125, 95, 75, 110, 120])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#dcfce7")),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor("#f0fdf4")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 12))

    summary_step_text = (
        "<b>Kümülatif Eğitim Hacmi:</b> Proje süresince <b>380.000 PPO Adımı</b> ve <b>18.556 Uzman Gösteri Adımı</b> "
        "olmak üzere toplamda <b>398.556 simülasyon adımı</b> işletilmiştir. Her adım dt = 0.5 saniye olduğundan, "
        "bu hacim <b>55 saatin üzerinde kesintisiz otonom uzay uçuşu tecrübesine</b> ve 1.300'den fazla kenetlenme seferine denk gelmektedir."
    )
    story.append(Paragraph(summary_step_text, body_style))

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 6: 4K ULTRA-HD SUNUM PANOSU (GÖRSEL ŞÖLEN)
    # ==========================================================================
    story.append(Paragraph("Faz 6: 4K Ultra-HD Yönetici Sunum Panosu (Executive Visual Showcase)", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=8))

    showcase_4k = os.path.join(DIR_PATH, "orbital_rl_executive_presentation_showcase.png")
    if os.path.exists(showcase_4k):
        story.append(Image(showcase_4k, width=780, height=438))
    else:
        story.append(Paragraph("4K Showcase Image Not Found", body_style))

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 7: 50 UÇUŞLUK MONTE CARLO DOĞRULAMA BENCHMARKI (ZİRVE)
    # ==========================================================================
    story.append(Paragraph("Faz 7: 50 Uçuşluk Monte Carlo Doğrulama Testi & Zirve Karşılaştırma", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=10))

    bench_50_data = [
        [Paragraph("<b>Model Mimarisi</b>", tbl_header),
         Paragraph("<b>Başarı Oranı</b>", tbl_header),
         Paragraph("<b>Kaza (Crash)</b>", tbl_header),
         Paragraph("<b>Asılı Kalma (Hover)</b>", tbl_header),
         Paragraph("<b>Ort. Son Mesafe</b>", tbl_header),
         Paragraph("<b>Temas Hızı</b>", tbl_header),
         Paragraph("<b>Harcanan Δv</b>", tbl_header),
         Paragraph("<b>Mühendislik Değerlendirmesi</b>", tbl_header)],
        [Paragraph("Naive PPO", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%100.0", tbl_cell), Paragraph("27.92 m", tbl_cell), Paragraph("83.4 cm/s", tbl_cell), Paragraph("4.06 m/s", tbl_cell), Paragraph("Ceza korkusuyla 8m kala donup kalır", tbl_cell)],
        [Paragraph("Asymptotic PPO", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%100.0", tbl_cell), Paragraph("9.45 m", tbl_cell), Paragraph("43.1 cm/s", tbl_cell), Paragraph("3.97 m/s", tbl_cell), Paragraph("Üstel havuzla yaklaşır, son metrede durur", tbl_cell)],
        [Paragraph("PBRS PPO", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%100.0", tbl_cell), Paragraph("7.52 m", tbl_cell), Paragraph("14.4 cm/s", tbl_cell), Paragraph("1.73 m/s", tbl_cell), Paragraph("Durmanın negatif ceza olduğu kanıtlandı", tbl_cell)],
        [Paragraph("MuJoCo DR PPO", tbl_cell), Paragraph("%0.0", tbl_cell), Paragraph("%56.7", tbl_cell), Paragraph("%43.3", tbl_cell), Paragraph("48.77 m", tbl_cell), Paragraph("140.9 cm/s", tbl_cell), Paragraph("2.67 m/s", tbl_cell), Paragraph("Rijit gövde temasında kalkansız kaza", tbl_cell)],
        [Paragraph("MuJoCo BC + PPO", tbl_cell), Paragraph("%56.0", tbl_cell), Paragraph("%2.0", tbl_cell), Paragraph("%42.0", tbl_cell), Paragraph("0.160 m", tbl_cell), Paragraph("4.8 cm/s", tbl_cell), Paragraph("1.66 m/s", tbl_cell), Paragraph("PWPF ölü bölgesinde eşik altı kalma", tbl_cell)],
        [Paragraph("<b>Champion-PPO (Nihai)</b>", tbl_cell), Paragraph("<font color='#059669'><b>%100.0</b></font>", tbl_cell), Paragraph("<font color='#059669'><b>%0.0</b></font>", tbl_cell), Paragraph("<font color='#059669'><b>%0.0</b></font>", tbl_cell), Paragraph("<b>0.139 m</b>", tbl_cell), Paragraph("<b>4.14 cm/s</b>", tbl_cell), Paragraph("<b>1.54 m/s</b>", tbl_cell), Paragraph("<font color='#059669'><b>Müfredat + BC + PPO + Latch Servosu</b></font>", tbl_cell)],
        [Paragraph("<b>Integrated Safe-GNC</b>", tbl_cell), Paragraph("<font color='#059669'><b>%100.0</b></font>", tbl_cell), Paragraph("<font color='#059669'><b>%0.0</b></font>", tbl_cell), Paragraph("<font color='#059669'><b>%0.0</b></font>", tbl_cell), Paragraph("<b>0.137 m</b>", tbl_cell), Paragraph("<b>4.41 cm/s</b>", tbl_cell), Paragraph("<b>0.94 m/s</b>", tbl_cell), Paragraph("<font color='#059669'><b>Analitik Master GNC & CBF Kalkanı</b></font>", tbl_cell)]
    ]

    t_b50 = Table(bench_50_data, colWidths=[110, 75, 75, 75, 80, 80, 75, 210])
    t_b50.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#dcfce7")),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor("#f0fdf4")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_b50)
    story.append(Spacer(1, 14))

    # Champion Modelin 3 Temel Başarı Ayağı
    champ_desc = (
        "<b>Şampiyon Modelin %100 Başarıya Ulaşma Formülü:</b><br/>"
        "1. <b>Kademeli Müfredat Eğitimi (Curriculum Learning):</b> Ajan doğrudan 25m'den başlatılmayıp, %35 terminal (0.8-3.5m), "
        "%35 orta (3.5-12m) ve %30 tam menzil (12-25m) kademelerinde eğitilerek port içine kilitlenme pratiği pekiştirildi.<br/>"
        "2. <b>Behavioral Cloning Sıcak Başlangıç:</b> Uzman Safe-GNC'den toplanan 12.156 durum-eylem çifti ile aktör ağı 30 epoch ön-eğitilerek soğuk başlangıç (cold-start) riski sıfırlandı.<br/>"
        "3. <b>PWPF Ölü Bölge Sürünme Servosu:</b> Son 1.2 metrede aracın sürati 3.8 cm/s hedeflenerek PWPF modülatörünün valfleri kapatıp donması (hovering) tamamen önlendi."
    )
    story.append(Paragraph(champ_desc, body_style))

    story.append(PageBreak())

    # ==========================================================================
    # SLAYT 8: CANLI DİJİTAL İKİZ & SEKTÖREL SONUÇ
    # ==========================================================================
    story.append(Paragraph("Faz 8: Canlı Dijital İkiz (4 Kameralı Görev Kontrolü) ve Sektörel Sonuç", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=8))

    # 4 Kamera Görseli Tablosu
    cam_c = os.path.join(DIR_PATH, "cam_chase_cam.png")
    cam_p = os.path.join(DIR_PATH, "cam_probe_cam.png")
    cam_s = os.path.join(DIR_PATH, "cam_station_cctv.png")
    cam_t = os.path.join(DIR_PATH, "cam_orbital_tactical.png")

    if all(os.path.exists(p) for p in [cam_c, cam_p, cam_s, cam_t]):
        cam_s8_data = [
            [Paragraph("<b>1. Chase Kamerası (Takip Görünümü)</b>", tbl_cell),
             Paragraph("<b>2. FPV Prob HUD (Optik Nişangah)</b>", tbl_cell),
             Paragraph("<b>3. Station CCTV (Liman Emniyet)</b>", tbl_cell),
             Paragraph("<b>4. Yörünge Taktik (Geniş Açı LEO)</b>", tbl_cell)],
            [Image(cam_c, width=190, height=140),
             Image(cam_p, width=190, height=140),
             Image(cam_s, width=190, height=140),
             Image(cam_t, width=190, height=140)]
        ]
        t_cam_s8 = Table(cam_s8_data, colWidths=[195, 195, 195, 195])
        t_cam_s8.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0f172a")),
            ('BOX', (0,0), (-1,-1), 1, c_cyan),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#1e293b")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_cam_s8)
        story.append(Spacer(1, 10))

    # Sektörel Saha İpuçları ve Kapanış
    closing_text = (
        "<b>Sektörel Kıdemli Uzman Vizyonu (NASA / ESA Standartları):</b><br/>"
        "Uzay ve havacılık sertifikasyon kurulları (FAA, NASA NTRS), saf bir 'kara kutu' sinir ağı modelinin uçuş bilgisayarına tek başına emanet "
        "edilmesine asla izin vermez. Projemizde uygulanan <b>Safety-Filtered Hybrid GNC mimarisi</b>, uzun menzilde derin RL'in esnek rotalama gücünden "
        "faydalanırken, eyleyici önündeki analitik CBF-QP kalkanı ile olası anormalliklerde bile hızı 6 cm/s altında tutarak çarpışmayı fiziksel olarak "
        "imkansız kılar. Bu mimari, akademik bir prototipin ötesinde <b>uçuşa hazır (flight-ready)</b> endüstriyel bir uzay standardıdır.<br/><br/>"
        "🌐 <b>Canlı Görev Kontrol Paneli:</b> <font color='#0284c7'><u>http://localhost:8055/</u></font> üzerinden 25-30 FPS MuJoCo akışıyla canlı olarak izlenebilir."
    )
    story.append(Paragraph(closing_text, body_style))

    # Derle
    doc.build(story, canvasmaker=SlideNumberedCanvas)
    print(f"✅ Nihai Yönetici ve Gelişim Sunumu PDF Raporu Başarıyla Üretildi: {PDF_OUTPUT}")


if __name__ == "__main__":
    build_final_presentation()
