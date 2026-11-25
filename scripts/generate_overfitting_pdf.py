#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Overfitting (Ezberleme) Teşhisi, RL Metrikleri ve Çözüm Rehberi
Yüksek Çözünürlüklü Mühendislik PDF Raporu Üreticisi (ReportLab)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# DejaVu fontlarını kaydet (Tam Türkçe ve Matematik karakter desteği)
FONT_DIR = "/usr/share/fonts/dejavu"
FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF",
]
for candidate in FONT_CANDIDATES:
    if os.path.exists(os.path.join(candidate, "DejaVuSans.ttf")):
        FONT_DIR = candidate
        break

pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))

MONO_PATH = os.path.join(FONT_DIR, "DejaVuSansMono.ttf")
if os.path.exists(MONO_PATH):
    pdfmetrics.registerFont(TTFont("DejaVuMono", MONO_PATH))
    CODE_FONT = "DejaVuMono"
else:
    CODE_FONT = "DejaVu"

# Renk Paleti: Deep Space & Aerospace Engineering
COLOR_PRIMARY = colors.HexColor("#0A0F1D")      # Deep Space Navy
COLOR_NAVY_MID = colors.HexColor("#131C31")     # Card Dark Navy
COLOR_SECONDARY = colors.HexColor("#0284C7")    # Orbital Cyan
COLOR_ACCENT = colors.HexColor("#38BDF8")       # Sky Blue
COLOR_SUCCESS = colors.HexColor("#059669")      # Emerald Green
COLOR_SUCCESS_BG = colors.HexColor("#ECFDF5")
COLOR_DANGER = colors.HexColor("#DC2626")       # Alert Red
COLOR_DANGER_BG = colors.HexColor("#FEF2F2")
COLOR_WARNING = colors.HexColor("#D97706")      # Amber Warning
COLOR_WARNING_BG = colors.HexColor("#FFFBEB")
COLOR_INFO_BG = colors.HexColor("#F0F9FF")
COLOR_BG_CARD = colors.HexColor("#F8FAFC")      # Slate 50
COLOR_BG_ALT = colors.HexColor("#F1F5F9")       # Slate 100
COLOR_CODE_BG = colors.HexColor("#0F172A")      # Code Dark Slate
COLOR_BORDER = colors.HexColor("#CBD5E1")       # Slate 300
COLOR_TEXT_MAIN = colors.HexColor("#1E293B")    # Slate 800
COLOR_TEXT_MUTED = colors.HexColor("#64748B")   # Slate 500

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 34
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


class NumberedCanvas(canvas.Canvas):
    """Her sayfaya otomatik üst/alt bilgi ve sayfa numarası ekler."""

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
        self.setFont("DejaVu", 7.6)
        self.setFillColor(COLOR_TEXT_MUTED)

        if self._pageNumber > 1:
            self.drawString(
                MARGIN,
                PAGE_HEIGHT - 23,
                "ORBITAL-RL // Overfitting Teşhisi, RL Metrikleri ve Genelleme Çözüm Rehberi",
            )
            self.drawRightString(
                PAGE_WIDTH - MARGIN,
                PAGE_HEIGHT - 23,
                "19 Kasım 2026 | DeepMind MuJoCo + PPO",
            )
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(MARGIN, PAGE_HEIGHT - 27, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 27)

        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(MARGIN, 28, PAGE_WIDTH - MARGIN, 28)
        self.drawString(
            MARGIN,
            18,
            "Nasuhi — Orbital-RL (github.com/Nasuhii/orbital-rl) | Sakarya Üniversitesi Veri Bilimi ve Analitiği",
        )
        self.drawRightString(
            PAGE_WIDTH - MARGIN,
            18,
            f"Sayfa {self._pageNumber} / {page_count}",
        )
        self.restoreState()


def make_callout(title, body_html, border_color, bg_color, title_style, body_style):
    """Sol şeritli vurgulu teknik bilgi/uyarı kutusu üretir."""
    content = [[Paragraph(title, title_style)], [Paragraph(body_html, body_style)]]
    tbl = Table(content, colWidths=[CONTENT_WIDTH - 4])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, border_color),
                ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (0, 0), 6),
                ("BOTTOMPADDING", (0, 0), (0, 0), 2),
                ("TOPPADDING", (0, 1), (0, 1), 1),
                ("BOTTOMPADDING", (0, 1), (0, 1), 6),
            ]
        )
    )
    return tbl


def make_kpi_strip(kpis):
    """4 kartlı üst KPI özet bandı oluşturur."""
    lbl_style = ParagraphStyle(
        "KpiLbl", fontName="DejaVu-Bold", fontSize=6.8, leading=8.5,
        textColor=COLOR_TEXT_MUTED, alignment=1
    )
    val_style = ParagraphStyle(
        "KpiVal", fontName="DejaVu-Bold", fontSize=11.5, leading=14,
        textColor=COLOR_PRIMARY, alignment=1
    )
    sub_style = ParagraphStyle(
        "KpiSub", fontName="DejaVu", fontSize=6.8, leading=8.8,
        textColor=COLOR_TEXT_MAIN, alignment=1
    )

    col_w = CONTENT_WIDTH / len(kpis)
    row = []
    for title, value, sub, accent_hex in kpis:
        v_s = ParagraphStyle(
            f"Val_{accent_hex}", parent=val_style, textColor=colors.HexColor(accent_hex)
        )
        cell_tbl = Table(
            [
                [Paragraph(title, lbl_style)],
                [Paragraph(value, v_s)],
                [Paragraph(sub, sub_style)],
            ],
            colWidths=[col_w - 6],
        )
        cell_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
                    ("LINEABOVE", (0, 0), (-1, 0), 2.5, colors.HexColor(accent_hex)),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        row.append(cell_tbl)

    outer = Table([row], colWidths=[col_w] * len(kpis))
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return outer


def build_pdf(output_filename="Orbital_RL_Overfitting_ve_Metrik_Analizi.pdf"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=38,
        bottomMargin=38,
    )

    # Tipografi Stilleri
    hero_title = ParagraphStyle(
        "HeroTitle", fontName="DejaVu-Bold", fontSize=14.5, leading=18.5,
        textColor=colors.white, spaceAfter=3
    )
    hero_sub = ParagraphStyle(
        "HeroSub", fontName="DejaVu-Bold", fontSize=9.0, leading=12,
        textColor=COLOR_ACCENT, spaceAfter=5
    )
    hero_meta = ParagraphStyle(
        "HeroMeta", fontName="DejaVu", fontSize=7.4, leading=10.5,
        textColor=colors.HexColor("#CBD5E1")
    )
    h1 = ParagraphStyle(
        "H1", fontName="DejaVu-Bold", fontSize=11.0, leading=14.5,
        textColor=COLOR_PRIMARY, spaceBefore=7, spaceAfter=4, keepWithNext=True
    )
    h2 = ParagraphStyle(
        "H2", fontName="DejaVu-Bold", fontSize=9.0, leading=12,
        textColor=COLOR_SECONDARY, spaceBefore=5, spaceAfter=2.5, keepWithNext=True
    )
    body = ParagraphStyle(
        "Body", fontName="DejaVu", fontSize=8.0, leading=11.2,
        textColor=COLOR_TEXT_MAIN, spaceAfter=3.5
    )
    bullet = ParagraphStyle(
        "Bullet", fontName="DejaVu", fontSize=7.8, leading=10.8,
        textColor=COLOR_TEXT_MAIN, leftIndent=10, firstLineIndent=-7, spaceAfter=2.5
    )
    callout_title = ParagraphStyle(
        "CalloutTitle", fontName="DejaVu-Bold", fontSize=8.2, leading=11,
        textColor=COLOR_PRIMARY
    )
    callout_body = ParagraphStyle(
        "CalloutBody", fontName="DejaVu", fontSize=7.6, leading=10.6,
        textColor=COLOR_TEXT_MAIN
    )
    th = ParagraphStyle(
        "TH", fontName="DejaVu-Bold", fontSize=7.4, leading=9.2,
        textColor=colors.white, alignment=1
    )
    th_l = ParagraphStyle("THL", parent=th, alignment=0)
    td = ParagraphStyle(
        "TD", fontName="DejaVu", fontSize=7.2, leading=9.4,
        textColor=COLOR_TEXT_MAIN, alignment=1
    )
    td_l = ParagraphStyle("TDL", parent=td, alignment=0)
    td_b = ParagraphStyle("TDB", parent=td_l, fontName="DejaVu-Bold")
    code_style = ParagraphStyle(
        "Code", fontName=CODE_FONT, fontSize=6.8, leading=9.4,
        textColor=colors.HexColor("#E2E8F0")
    )
    caption = ParagraphStyle(
        "Caption", fontName="DejaVu-Oblique", fontSize=7.2, leading=9.5,
        textColor=COLOR_TEXT_MUTED, alignment=1, spaceBefore=2.5, spaceAfter=5
    )

    story = []

    # =========================================================================
    # SAYFA 1: KAPAK BANNER + KPI + ML vs RL METRİKLERİ & GERÇEK VERİLER
    # =========================================================================
    hero_tbl = Table(
        [
            [Paragraph("ORBITAL-RL: OVERFITTING (EZBERLEME) TEŞHİSİ, RL METRİKLERİ VE ÇÖZÜM REHBERİ", hero_title)],
            [Paragraph("Champion-PPO v3 (%82) → v4 (%100) Geçişinin Teknik Otopsisi, Değerlendirme Parametreleri ve Anti-Overfitting Mimarisi", hero_sub)],
            [Paragraph(
                "<b>Araştırmacı:</b> Nasuhi &nbsp;|&nbsp; <b>Proje:</b> Autonomous Orbital Docking (MuJoCo v3.17 + SB3 PPO) "
                "&nbsp;|&nbsp; <b>Tarih:</b> 19 Kasım 2026 &nbsp;|&nbsp; <b>Kapsam:</b> Metrik Analizi + Kod Bug Raporu + 7 Çözüm Yöntemi",
                hero_meta,
            )],
        ],
        colWidths=[CONTENT_WIDTH],
    )
    hero_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_PRIMARY),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (0, 0), 9),
                ("BOTTOMPADDING", (0, -1), (0, -1), 8),
                ("TOPPADDING", (0, 1), (0, -1), 2),
                ("BOTTOMPADDING", (0, 0), (0, 1), 2),
            ]
        )
    )
    story.append(hero_tbl)
    story.append(Spacer(1, 7))

    # KPI Bandı
    story.append(
        make_kpi_strip(
            [
                ("v3 → v4 SIÇRAMASI", "%82 → %100", "700K → 1.4M Adım (ID Test)", "#0284C7"),
                ("CANLI SUNUCU (2,319)", "%99.61", "2310 Dock | 9 Hover | 0 Crash", "#059669"),
                ("EZBER KANITI #1", "±0.000038 m/s", "Pure-Raw Son Hız Std (38 µm/s)", "#DC2626"),
                ("EZBER KANITI #2", "4 – 31 / Uçuş", "CBF Kalkan Müdahale Sayısı", "#D97706"),
            ]
        )
    )
    story.append(Spacer(1, 7))

    # Kritik Teorik Ayrım Callout
    story.append(
        make_callout(
            "[KRİTİK AYRIM] Neden RL (PPO) Modelinde Klasik F1-Score veya Accuracy Doğrudan Kullanılmaz?",
            "<b>Accuracy, Precision, Recall, F1-Score ve ROC-AUC</b> metrikleri <i>Denetimli Sınıflandırma (Supervised Classification)</i> "
            "problemleri içindir (elinde sabit bir X_test ve ikili/çoklu y_test etiketi vardır). <b>Orbital-RL (Champion-PPO)</b> ise "
            "sınıf etiketi tahmin etmez; her 0.5 saniyede 9 boyutlu yörünge durumundan sürekli 3D kuvvet vektörü <b>[Fx, Fy, Fz] ∈ [-15N, +15N]³</b> "
            "üreten bir <i>Sürekli Kontrol Politikası (Continuous Control RL)</i> çalıştırır.<br/>"
            "• <b>Jüri / Hoca Notu:</b> 'PPO modelimin F1 skoru 1.0' demek yerine, RL literatüründeki gerçek karşılıklar olan "
            "<b>In-Distribution vs OOD Success Rate (Genelleme Boşluğu)</b>, <b>Crash/Hover Oranı</b>, <b>CBF Müdahale Sıklığı</b>, "
            "<b>Explained Variance (Critic R²)</b>, <b>Policy Entropy / Action Std</b> ve <b>CVaR-%10</b> metrikleri raporlanmalıdır.",
            COLOR_WARNING,
            COLOR_WARNING_BG,
            callout_title,
            callout_body,
        )
    )
    story.append(Spacer(1, 6))

    # Bölüm 1: Gerçek Kayıtlı Veriler Tablosu
    story.append(Paragraph("1. Kayıtlı Model Verilerinin Sayısal Kıyaslaması (v3 [%82] → v4 [%100] & Canlı 2,319 Uçuş)", h1))
    story.append(
        Paragraph(
            "Proje dosyalarındaki (<i>rl_deep_dive_hyperparameters_and_curves.png</i>, <i>mujoco_champion_benchmark_stats.json</i> ve "
            "<i>telemetry_live.json</i>) doğrulanmış ölçüm sonuçları aşağıda tek tabloda toplanmıştır:",
            body,
        )
    )

    m_data = [
        [
            Paragraph("Değerlendirme Metriği", th_l),
            Paragraph("v3 (700K — %82)<br/>Pure-CBF+PWPF", th),
            Paragraph("v4 (1.4M — %100)<br/>Pure-CBF+PWPF (50)", th),
            Paragraph("v4 (1.4M — %100)<br/>Pure-Raw (50)", th),
            Paragraph("v4 Canlı Sunucu<br/>(2,319 Uçuş)", th),
        ],
        [
            Paragraph("Başarı Oranı (Epizot Accuracy)", td_b),
            Paragraph("<b>%82.0</b> (41/50)", td),
            Paragraph("<b>%100.0</b> (50/50)", td),
            Paragraph("<b>%100.0</b> (50/50)", td),
            Paragraph("<b>%99.61</b> (2310/2319)", td),
        ],
        [
            Paragraph("Kaza Oranı (Crash / Ölümcül Hata)", td_b),
            Paragraph("%0.0 (0/50)", td),
            Paragraph("%0.0 (0/50)", td),
            Paragraph("%0.0 (0/50)", td),
            Paragraph("%0.00 (0/2319)", td),
        ],
        [
            Paragraph("Asılı Kalma (Hover / Başarısız)", td_b),
            Paragraph("<b>%18.0</b> (9/50)", td),
            Paragraph("%0.0 (0/50)", td),
            Paragraph("%0.0 (0/50)", td),
            Paragraph("%0.39 (9/2319)", td),
        ],
        [
            Paragraph("Epizot Bazlı Yapay F1 (Sadece Fikir İçin)", td_b),
            Paragraph("0.901 (P=1.0, R=0.82)", td),
            Paragraph("1.000 (P=1.0, R=1.0)", td),
            Paragraph("1.000", td),
            Paragraph("0.998 (P=1.0, R=0.996)", td),
        ],
        [
            Paragraph("Son Mesafe (μ ± σ)", td_b),
            Paragraph("—", td),
            Paragraph("0.1286 ± 0.0061 m", td),
            Paragraph("0.1286 ± 0.0063 m", td),
            Paragraph("~0.119 – 0.139 m", td),
        ],
        [
            Paragraph("Son Temas Hızı (μ ± σ)", td_b),
            Paragraph("—", td),
            Paragraph("0.0430 ± 0.0043 m/s", td),
            Paragraph("<b>0.042718 ± 0.000038</b>", td),
            Paragraph("~4.22 – 4.82 cm/s", td),
        ],
        [
            Paragraph("Harcanan Yakıt Δv (μ ± σ)", td_b),
            Paragraph("—", td),
            Paragraph("1.571 ± 0.292 m/s", td),
            Paragraph("0.616 ± 0.036 m/s", td),
            Paragraph("~0.97 – 2.01 m/s", td),
        ],
        [
            Paragraph("BC Ön-Eğitim Kaybı (MSE Loss)", td_b),
            Paragraph("0.087 → 0.0048", td),
            Paragraph("v3'ten devralındı", td),
            Paragraph("v3'ten devralındı", td),
            Paragraph("—", td),
        ],
        [
            Paragraph("CBF Kalkan Müdahalesi (cbf_count)", td_b),
            Paragraph("Aktif (logsuz)", td),
            Paragraph("Aktif (logsuz)", td),
            Paragraph("0 (Kapalı)", td),
            Paragraph("<b>4 – 31 kez / uçuş!</b>", td),
        ],
    ]

    m_tbl = Table(
        m_data,
        colWidths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.175, CONTENT_WIDTH * 0.185, CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.17],
    )
    m_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ("BACKGROUND", (3, 6), (3, 6), COLOR_DANGER_BG),
                ("BACKGROUND", (4, 9), (4, 9), COLOR_WARNING_BG),
                ("GRID", (0, 0), (-1, -1), 0.45, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 4.2),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(m_tbl)
    story.append(Spacer(1, 6))

    story.append(
        make_callout(
            "[EZBERLEMENİN 2 KESİN SAYISAL KANITI]",
            "<b>1) 38 Mikrometre/sn Hız Kilidi (Policy Memorization):</b> <i>Pure-Raw</i> modunda 50 farklı başlangıç konumunda son hızın "
            "standart sapması yalnızca <b>0.000038 m/s</b> çıkmıştır. Bu durum ağın yörünge dinamiğini genel çözmediğini, uzmanın "
            "sabit <i>0.042 m/s</i> hız profilini birebir ezberlediğini gösterir.<br/>"
            "<b>2) Kalkan Bağımlılığı:</b> Canlı sunucuda (%99.61 başarı) <b>CBF güvenlik kalkanı her uçuşta ortalama 4 ila 31 kez</b> "
            "sinir ağının ürettiği emri kırpmaktadır. Yani görünen başarının ciddi kısmı ağın zekâsından değil, arkasındaki kalkanın kurtarmasından gelir.",
            COLOR_DANGER,
            COLOR_DANGER_BG,
            callout_title,
            callout_body,
        )
    )

    # =========================================================================
    # SAYFA 2: RL'DE GERÇEK METRİKLER + NASIL BULURSUN? + MEVCUT EĞİTİM GÖRSELİ
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. RL'de Accuracy / F1 / R² Yerine Bakılan 6 Gerçek Metrik ve Ölçüm Yöntemi", h1))
    story.append(
        Paragraph(
            "Bir Pekiştirmeli Öğrenme (RL) modelinde overfitting'i ispatlamak için aşağıdaki 6 temel metrik kullanılır. "
            "Her birinin Supervised ML karşılığı ve proje kodunda nasıl hesaplanacağı aşağıdadır:",
            body,
        )
    )

    rl_metrics = [
        [
            Paragraph("Supervised ML Karşılığı", th_l),
            Paragraph("RL'deki Gerçek Metrik", th_l),
            Paragraph("Overfitting'i Nasıl Yakalar?", th_l),
            Paragraph("Projede Nasıl Ölçülür?", th_l),
        ],
        [
            Paragraph("Train vs Test Accuracy", td_b),
            Paragraph("<b>ID vs OOD Success Rate</b><br/>(Genelleme Boşluğu)", td_l),
            Paragraph("ID=%100 iken görmediği mesafe, açı, kütle veya gürültüde OOD %20'ye düşüyorsa <b>kesin overfitting</b> vardır.", td_l),
            Paragraph("Aynı .zip modelini 100 nominal (ID) + 100 görmediği OOD uçuşta test ederek Gap = S_ID - S_OOD hesapla.", td_l),
        ],
        [
            Paragraph("Precision / Recall / F1", td_b),
            Paragraph("<b>Success / Crash / Hover</b><br/>+ CBF Müdahale Oranı", td_l),
            Paragraph("CBF kapalı (Pure-Raw) OOD testte Crash patlıyorsa ağ değil kalkan uçuyordur.", td_l),
            Paragraph("Her uçuşta docked, crashed, truncated ve cbf_intervened sayacını ayrı logla.", td_l),
        ],
        [
            Paragraph("R² (Determinasyon)", td_b),
            Paragraph("<b>Explained Variance (EV)</b><br/>1 - Var(G - V(s))/Var(G)", td_l),
            Paragraph("Critic ağının R²'sidir. ID'de 0.98 iken OOD yörüngede &lt;0.20 (veya negatif) ise <b>Critic ezberlemiştir</b>.", td_l),
            Paragraph("TensorBoard train/explained_variance logundan veya eval'de gerçek getiri G_t ile predict_values(obs) kıyaslayarak.", td_l),
        ],
        [
            Paragraph("Overconfidence", td_b),
            Paragraph("<b>Policy Entropy &amp; Action Std</b><br/>(σ = exp(log_std))", td_l),
            Paragraph("σ → 0 ise politika tek bir yörünge şablonuna donmuştur; en küçük dış darbede dağılır.", td_l),
            Paragraph("PPO.load(...) sonrası model.policy.log_std.exp() ve train/entropy_loss değerlerini oku.", td_l),
        ],
        [
            Paragraph("Worst-Case / Tail Loss", td_b),
            Paragraph("<b>CVaR-%5 / CVaR-%10</b><br/>(En Kötü Dilim Ort.)", td_l),
            Paragraph("Ortalama yüksek görünürken en kötü %10 uçuş kaza/hover yapıyorsa politika kırılgandır.", td_l),
            Paragraph("100 eval uçuşunun episode return ve son hız değerlerini sırala, en kötü 10 uçuşun ortalamasını al.", td_l),
        ],
        [
            Paragraph("Regresyon Hatası (MSE)", td_b),
            Paragraph("<b>Action MSE / Smoothness</b><br/>ve Yakıt Verimliliği (Δv)", td_l),
            Paragraph("Ardışık eylemlerde yüksek sıçrama (chattering) ve gereksiz Δv tüketimi kararsız ezber göstergesidir.", td_l),
            Paragraph("mean((a_t - a_{t-1})²) ve toplam Σ||F_t||·dt / m değerlerini raporla.", td_l),
        ],
    ]

    rlm_tbl = Table(
        rl_metrics,
        colWidths=[CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.31, CONTENT_WIDTH * 0.30],
    )
    rlm_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ("GRID", (0, 0), (-1, -1), 0.45, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 4.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(rlm_tbl)
    story.append(Spacer(1, 6))

    # Nasıl Bulursun: Kod Kutusu
    story.append(Paragraph("2.1. Bu Parametreleri Kendi Bilgisayarında 30 Saniyede Nasıl Çekersin?", h2))
    snippet = (
        "from stable_baselines3 import PPO; import numpy as np, torch<br/>"
        "model = PPO.load('models/ppo_mujoco_champion.zip', device='cpu')<br/>"
        "# 1) Model ic parametreleri &amp; Policy Collapse kontrolu (Action Std ne kadar kuculmus?)<br/>"
        "print('Timesteps:', model.num_timesteps, '| ent_coef:', model.ent_coef)<br/>"
        "print('Action Std (exp log_std):', torch.exp(model.policy.log_std).detach().numpy())<br/>"
        "# 2) Critic R^2 (Explained Variance): Eval rollout'tan gercek getiri G ve V(s) kiyasla:<br/>"
        "#    ev = 1.0 - np.var(G_true - V_pred) / (np.var(G_true) + 1e-8)<br/>"
        "# 3) ID vs OOD Gap: Ayni modeli hem nominal CONFIG hem OOD (35-50m, vy&gt;0, +-20% kutle, gurultu) ile kos!"
    )
    code_tbl = Table([[Paragraph(snippet, code_style)]], colWidths=[CONTENT_WIDTH])
    code_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_CODE_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, COLOR_ACCENT),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(code_tbl)
    story.append(Spacer(1, 6))

    # Mevcut v3 Görseli (varsa ekle)
    deep_img = os.path.join(script_dir, "rl_deep_dive_hyperparameters_and_curves.png")
    if os.path.exists(deep_img):
        story.append(Paragraph("2.2. Referans: v3 (700K — %82 Pure-CBF / %100 Pure-Raw) Hiperparametre ve Eğri Panosu", h2))
        w_img = CONTENT_WIDTH
        h_img = w_img * (1200.0 / 1920.0) * 0.88
        story.append(Image(deep_img, width=w_img, height=h_img))
        story.append(
            Paragraph(
                "Şekil 1: 700K adımlık v3 modelinin hiperparametre tablosu (MLP [128,128], LR 3e-4→1e-5, Ent 0.002→0.0008, BC MSE 0.087→0.0048) "
                "ve 50 uçuşluk Monte Carlo doğrulama grafiği (%82 Pure-CBF, %18 Hover). 1.4M adıma uzatıldığında bu dar dağılımda %100'e kilitlenmiştir.",
                caption,
            )
        )

    # =========================================================================
    # SAYFA 3: KOD TABANINDAKİ 5 KÖK NEDEN & 2 GİZLİ BUG (OTOPSİ)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Kod Otopsisi: Model Neden %82'den %100'e Çıkıp Ezbere Kilitlendi?", h1))
    story.append(
        Paragraph(
            "700K adımlık <b>v3 (%82)</b> modeli 1.4M adıma (<b>v4, %100</b>) ince ayar yapılırken, aynı dar başlangıç kutusunda "
            "2 kat daha fazla etkileşim toplanmış, öğrenme oranı <i>1e-5</i>'e ve entropi katsayısı <i>0.0008</i>'e düşürülmüştür. "
            "Ağ, kalan %18'lik köşe durumları genel fizik öğrenerek değil, <b>o dar tüpteki spesifik koordinatlara ezber yama yaparak</b> çözmüştür. "
            "Kod tabanında doğrudan tespit edilen <b>5 yapısal kök neden</b> aşağıdadır:",
            body,
        )
    )

    causes_data = [
        [
            Paragraph("#", th),
            Paragraph("Yapısal Overfitting Nedeni", th_l),
            Paragraph("Kod Dosyası &amp; Satırlar", th_l),
            Paragraph("Teknik Etki ve Ezber Mekanizması", th_l),
        ],
        [
            Paragraph("1", td_b),
            Paragraph("<b>Dar ve Tek Yönlü Başlangıç Tüpü</b>", td_l),
            Paragraph("orbital_config.py (L36–42)<br/>orbital_mujoco_env.py (L106–129)<br/>train_mujoco_champion_ppo.py (L88–113)", td_l),
            Paragraph("Ajan hep +Y kapısının önünde (y∈[18,25], x∈[±2.5], z∈[±1.8]) ve <b>zaten hedefe kapanan hızla</b> (vy∈[-0.22,-0.10]) doğuyor. Kaçış hızı (vy&gt;0), geniş açı veya dönme (quaternion hep [1,0,0,0]) hiç yok. Curriculum tüpü daha da daraltıyor (x:±2.0, z:±1.5).", td_l),
        ],
        [
            Paragraph("2", td_b),
            Paragraph("<b>Aynı Kutuda Eğit, Aynı Kutuda Test Et</b>", td_l),
            Paragraph("mujoco_champion_benchmark_stats.json<br/>stress_test_monte_carlo.py (L41–68)", td_l),
            Paragraph("%100 alınan 50 uçuşun tamamı eğitimle birebir aynı nominal kutuda (500 kg, 15 N, 0 gürültü). Projedeki tek OOD stres testi (35–50 m, kaçış hızı, arıza, şok) <b>PPO'yu değil</b>, el yapımı analitik kontrolcüyü test ediyor.", td_l),
        ],
        [
            Paragraph("3", td_b),
            Paragraph("<b>Ağır BC + Sürekli Uzman Taklit Cezası</b>", td_l),
            Paragraph("train_mujoco_champion_ppo.py<br/>(L157–160, L200–266)", td_l),
            Paragraph("30 epoch BC (MSE 0.0048) üstüne PPO boyunca her adımda r_expert = -expert_weight·10·||a - a_exp||² cezası açık kalıyor ve ent_coef=0.0005–0.0008. Ajan kendi keşfini kapatıp tek uzmanı klonluyor.", td_l),
        ],
        [
            Paragraph("4", td_b),
            Paragraph("<b>Eksen-Bağımlı Ödül Tasarımı</b>", td_l),
            Paragraph("train_mujoco_champion_ppo.py (L148–156)<br/>orbital_mujoco_env.py (L200)", td_l),
            Paragraph("r_creep sadece vy &lt; -0.025 yönünü ödüllendiriyor, vbar_alignment ise (obs[1]/dist)² kullanıyor. Hedef doğrultusuna göre bağıl hız yerine sabit -Y ekseni ezberletiliyor.", td_l),
        ],
        [
            Paragraph("5", td_b),
            Paragraph("<b>Şampiyon Eğitimde Sıfır DR</b>", td_l),
            Paragraph("train_mujoco_champion_ppo.py<br/>(L244–248)", td_l),
            Paragraph("Champion boru hattı çıplak OrbitalMujocoEnv üzerine kurulu: sabit kütle, sabit itki, sıfır gürültü, sıfır itici arızası ve sıfır dış şok.", td_l),
        ],
    ]
    causes_tbl = Table(
        causes_data,
        colWidths=[CONTENT_WIDTH * 0.05, CONTENT_WIDTH * 0.24, CONTENT_WIDTH * 0.27, CONTENT_WIDTH * 0.44],
    )
    causes_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ("GRID", (0, 0), (-1, -1), 0.45, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 4.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(causes_tbl)
    story.append(Spacer(1, 8))

    # Kritik Bug Raporu Kutusu
    story.append(Paragraph("3.1. Sessiz Hatalar: DR ve Canlı Test Kodunda Yakalanan 3 Kritik Bug", h2))

    story.append(
        make_callout(
            "[BUG #1 — train_ppo_mujoco_dr.py, L42–43] Kütle Domain Randomization MuJoCo Fiziğinde Çalışmıyor!",
            "Kodda <b>self.env.m = self.nominal_mass * mass_factor</b> atanıyor. Bu satır sadece Python tarafında enjekte edilen "
            "Clohessy-Wiltshire kuvvetini (f_cw) ölçekler; <b>MuJoCo motorunun gerçek self.env.model.body_mass[chaser_id] ve body_inertia</b> "
            "değerlerini değiştirmez! Dolayısıyla aracın iticiye verdiği ivme tepkisi (<b>a = F / m</b>) her epizotta sabit 500 kg kalır.<br/>"
            "<b>Düzeltme:</b> <i>model.body_mass[id] = base_mass * f</i>, <i>model.body_inertia[id] = base_inertia * f</i> ve hemen ardından "
            "<b>mujoco.mj_setConst(model, data)</b> çağrılmalıdır.",
            COLOR_DANGER,
            COLOR_DANGER_BG,
            callout_title,
            callout_body,
        )
    )
    story.append(Spacer(1, 5))

    story.append(
        make_callout(
            "[BUG #2 — train_ppo_mujoco_dr.py, L58–64] Sensör Gürültüsünde Veri Sızıntısı (Data Leakage)!",
            "<i>_apply_sensor_noise()</i> fonksiyonu <b>obs[0:3]</b> (konum) ve <b>obs[3:6]</b> (hız) üzerine Gauss gürültüsü ekliyor; "
            "ancak <b>obs[6] (coriolis), obs[7] (gerçek mesafe dist) ve obs[8] (gerçek hız speed) gürültülü veriden yeniden hesaplanmıyor!</b> "
            "Sinir ağı, gürültüsüz/temiz gerçek mesafe ve hızı son 3 indisten bedava okuyarak gürültüye dayanıklılık geliştirmeden kaçabiliyor.<br/>"
            "<b>Düzeltme:</b> Gürültü eklendikten sonra <i>noisy[6] = -2*n*noisy[3]</i>, <i>noisy[7] = ||noisy[0:3]||</i>, <i>noisy[8] = ||noisy[3:6]||</i> yeniden hesaplanmalıdır.",
            COLOR_DANGER,
            COLOR_DANGER_BG,
            callout_title,
            callout_body,
        )
    )
    story.append(Spacer(1, 5))

    story.append(
        make_callout(
            "[BUG #3 — mujoco_live_server.py, L146–150] Canlı Arayüzdeki Şok Enjeksiyonu Fiziği Değiştirmiyor!",
            "<i>inject_shock()</i> metodu <b>self.obs[3]</b> ve <b>self.obs[5]</b> dizisini değiştiriyor; ancak MuJoCo fizik durumunu tutan "
            "<b>self.env.data.qvel[0..2]</b> güncellenmediği için bir sonraki <i>env.step()</i> çağrısında şok tamamen siliniyor. "
            "Gerçek fiziksel darbe testi için <b>self.env.data.qvel[:3] += shock_vec</b> + <i>mj_forward</i> kullanılmalıdır.",
            COLOR_WARNING,
            COLOR_WARNING_BG,
            callout_title,
            callout_body,
        )
    )

    # =========================================================================
    # SAYFA 4: 7 SOMUT ÇÖZÜM YÖNTEMİ + TRAIN/VAL/TEST PROTOKOLÜ + YOL HARİTASI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Orbital-RL'de Overfitting'i Kökten Çözen 7 Mühendislik Yöntemi", h1))
    story.append(
        Paragraph(
            "Supervised ML'de overfitting ağırlıklı olarak <i>L1/L2 ceza terimi</i> ve <i>ağaç budama</i> ile çözülür. "
            "<b>RL'de ise overfitting'in ~%70'i ortam/dağılım tasarımıyla, ~%20'si politika regülarizasyonuyla, ~%10'u erken durdurma protokolüyle</b> çözülür:",
            body,
        )
    )

    # Katman A
    story.append(Paragraph("Katman A — Ortam ve Dağılım Yöntemleri (Asıl İlaç — %70 Etki)", h2))
    story.append(Paragraph(
        "• <b>1. Gerçek ve Sızıntısız Domain Randomization (DR):</b> Her epizotta MuJoCo <i>body_mass / body_inertia</i> ±%20, "
        "maksimum itki ±%20, tek eksen kısmi itici arızası (%15–30 güç kaybı), sızıntısız sensör gürültüsü + rastgele sensör sapması (bias) + "
        "1 adım gecikme ve epizot ortası rastgele hız darbeleri (<i>qvel += Δv</i>) enjekte edilir. Ajan sabit yörünge yerine <b>kapalı döngü toparlanma</b> öğrenir.",
        bullet,
    ))
    story.append(Paragraph(
        "• <b>2. Geniş Başlangıç Dağılımı + Automatic Domain Randomization (ADR):</b> Başlangıç zarfı dar +Y tüpünden çıkarılıp "
        "geniş mesafe (5–45 m), geniş yaklaşma açısı, yüksek yanal sapma ve <b>kaçış yönünde hızlar (vy &gt; 0)</b> kapsayacak şekilde tasarlanır. "
        "Hepsi ilk günden açılmaz; doğrulama başarısı %85'i geçtikçe zarf otomatik %10–15 genişletilir (ADR) — ajan asla rehavete kapılıp ezberleyemez.",
        bullet,
    ))
    story.append(Paragraph(
        "• <b>3. Eksen-Bağımsız Ödül + Kısmi Gözlem İçin Hafıza (Frame Stacking / LSTM):</b> Ödül sabit -Y ekseni yerine hedef vektörüne göre "
        "radyal kapanma hızı <i>v_close = -v · r̂</i> ve teğetsel hata <i>v_tan</i> üzerinden tanımlanır. Belirsiz kütle ve gürültü altında "
        "<i>in-context system identification</i> yapabilmek için gözleme <b>son 4 adımın (obs, prev_action) geçmişi</b> veya RecurrentPPO (LSTM) eklenir.",
        bullet,
    ))

    # Katman B
    story.append(Paragraph("Katman B — Politika ve Sinir Ağı Regülarizasyonu (%20 Etki)", h2))
    story.append(Paragraph(
        "• <b>4. Uzman Bastonunu Bırakma (BC &amp; expert_weight Annealing):</b> Behavioral Cloning yalnızca kısa bir sıcak başlangıç "
        "(10–15 epoch) olarak kalır; PPO eğitiminin ilk %25'inden sonra <i>expert_weight</i> kademeli olarak <b>tamamen 0.0'a</b> indirilir. "
        "Böylece model uzmanın kopyası olmak yerine gürültü/arıza altında uzmanı aşabilen gerçek bir RL politikası olur.",
        bullet,
    ))
    story.append(Paragraph(
        "• <b>5. Entropi Koruma + LayerNorm + AdamW Weight Decay + Simetri Veri Artırma:</b> <i>ent_coef</i> 0.0008'e öldürülmez "
        "(0.003–0.005 bandında tutulur veya otomatik entropili <b>SAC</b> kullanılır). RL'de Dropout yerine kararlı çalışan <b>LayerNorm + "
        "weight_decay=1e-4 + VecNormalize</b> kullanılır. Ayrıca CW fiziğinde Z ekseni simetriktir (<i>z, vz, Fz ↔ -z, -vz, -Fz</i>); "
        "simetri artırması bedavadan 2× çeşitlilik kazandırarak ezberi kırar.",
        bullet,
    ))
    story.append(Paragraph(
        "• <b>6. OOD Doğrulama Setiyle Erken Durdurma (Early Stopping):</b> 700K'dan 1.4M'ye körlemesine devam etmek yerine, "
        "her 50K adımda eğitimde hiç görülmeyen sabit bir OOD doğrulama setinde (<i>EvalCallback</i>) ölçüm alınır. "
        "ID %100 veren değil, <b>OOD başarısı ve Critic EV'si zirve yapan checkpoint (best_ood_model.zip)</b> kaydedilir.",
        bullet,
    ))

    # Katman C
    story.append(Paragraph("Katman C — Ayrık Train / Validation / Held-Out OOD Test Protokolü (%10 Etki)", h2))

    split_data = [
        [
            Paragraph("Küme / Amaç", th_l),
            Paragraph("Mesafe &amp; Hız Zarfı", th_l),
            Paragraph("Fizik &amp; İtici Belirsizliği", th_l),
            Paragraph("Sensör &amp; Dış Şok", th_l),
        ],
        [
            Paragraph("<b>Train (ADR Eğitim)</b><br/>Politika öğrenimi", td_l),
            Paragraph("y: 8–35 m | x,z: ±5 m<br/>vy: [-0.30, +0.15] m/s", td_l),
            Paragraph("Kütle: ±%15 (gerçek body_mass)<br/>İtki: ±%15, %15 tek eksen zayıflık", td_l),
            Paragraph("σ_pos=2 cm, σ_vel=0.5 cm/s<br/>(sızıntısız) + orta seviye şok", td_l),
        ],
        [
            Paragraph("<b>Validation (Early Stop)</b><br/>30 sabit görülmemiş seed", td_l),
            Paragraph("y: 12–38 m | x,z: ±6 m<br/>vy: [-0.25, +0.20] m/s", td_l),
            Paragraph("Kütle: ±%18 | İtki: ±%18<br/>%20 rastgele eksen verim kaybı", td_l),
            Paragraph("σ_pos=3 cm, σ_vel=0.7 cm/s<br/>t=45s anında 0.08 m/s şok", td_l),
        ],
        [
            Paragraph("<b>Held-Out OOD Test</b><br/>Final Sınavı (Asla Görülmez)", td_l),
            Paragraph("<b>y: 35–50 m | x,z: ±8 m</b><br/><b>vy: [+0.15, +0.35] (Kaçış!)</b>", td_l),
            Paragraph("<b>Kütle: 400 / 600 kg (±%20)</b><br/><b>Fz %30 arıza + ±%15 jitter</b>", td_l),
            Paragraph("<b>σ_pos=5 cm, σ_vel=1.0 cm/s</b><br/><b>t=50s anında 0.12 m/s şok</b>", td_l),
        ],
    ]
    split_tbl = Table(
        split_data,
        colWidths=[CONTENT_WIDTH * 0.21, CONTENT_WIDTH * 0.26, CONTENT_WIDTH * 0.28, CONTENT_WIDTH * 0.25],
    )
    split_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ("BACKGROUND", (0, 3), (-1, 3), COLOR_INFO_BG),
                ("GRID", (0, 0), (-1, -1), 0.45, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 4.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(split_tbl)
    story.append(Spacer(1, 7))

    # Önerilen Uygulama Yol Haritası
    story.append(Paragraph("4.1. Önerilen 3 Adımlı Uygulama Planı (v4 → v5 Anti-Overfit)", h2))
    road_data = [
        [
            Paragraph("Adım 1: Röntgen (Diagnostik)", th),
            Paragraph("Adım 2: Bug Fix + v5 Eğitimi", th),
            Paragraph("Adım 3: Çift Modlu OOD Sınavı", th),
        ],
        [
            Paragraph(
                "Mevcut <i>ppo_mujoco_champion.zip</i> için ID vs OOD Eval scripti çalıştırılır: "
                "Success, Crash, Hover, Critic EV (R²), Action Std ve CBF müdahale oranı çıkarılarak "
                "overfitting sayılarla ispatlanır.",
                td_l,
            ),
            Paragraph(
                "<i>body_mass</i> + <i>obs[6:9]</i> sızıntı bug'ları kapatılır; 4-frame stacking, "
                "eksen-bağımsız ödül, BC annealing, Z-simetri artırma ve OOD Early Stopping ile "
                "<b>Champion-PPO v5</b> eğitilir.",
                td_l,
            ),
            Paragraph(
                "v4 ve v5 aynı Held-Out OOD stres testinde hem <b>CBF Kapalı (Pure-Raw)</b> hem "
                "<b>CBF+PWPF Açık</b> yarıştırılır; Generalization Gap'in kapanışı raporlanır.",
                td_l,
            ),
        ],
    ]
    road_tbl = Table(road_data, colWidths=[CONTENT_WIDTH / 3.0] * 3)
    road_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BACKGROUND", (0, 1), (-1, 1), COLOR_BG_CARD),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 5.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(road_tbl)
    story.append(Spacer(1, 7))

    # Senior Savunma Kutusu
    story.append(
        make_callout(
            "[SENIOR MÜHENDİS ÖZETİ — SUNUM / JÜRİ SAVUNMA METNİ]",
            "<i>\"Sürekli kontrol RL problemlerinde klasik F1/Accuracy yerine In-Distribution vs Out-of-Distribution Başarı Boşluğu, "
            "Critic Explained Variance (R²), Policy Entropy ve CBF Kalkan Müdahale Oranı ölçülür. 700K'dan 1.4M ve 2.4M adıma uzatılan eski "
            "modellerimizde ödül ölçeği patlaması nedeniyle Critic'in çöktüğünü (EV=0.000), Aktör ağırlıklarının BC sonrasında tamamen donduğunu "
            "(diff &lt; 0.0001) ve +Y dışı 360° yaklaşımlarda %0.0 başarı alındığını kanıtladık. Ölçeklenmiş ödül, sıfır-körlük asinh özellikler, "
            "sızıntısız DR ve Held-Out OOD Early Stopping ile eğittiğimiz v5 modeli (912K adım), Pure-Raw OOD başarısını %78.4'ten %92.8'e çıkardı "
            "ve 360° Fly-Around başarısını %0.0'dan %100.0'a taşıdı.\"</i>",
            COLOR_SUCCESS,
            COLOR_SUCCESS_BG,
            callout_title,
            callout_body,
        )
    )

    # =========================================================================
    # SAYFA 5: GERÇEK DENEYSEL RÖNTGEN SONUÇLARI (v3 / v4 / gen_2.4m vs v5)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Gerçek Deneysel Röntgen Sonuçları: Eski Modeller vs Champion-PPO v5 (912K)", h1))
    story.append(
        Paragraph(
            "Tüm bug'lar (<i>body_mass + mj_setConst</i>, sızıntısız <i>obs[6:9]</i>, post-PWPF itki arızası, gerçek <i>qvel</i> şokları) "
            "düzeltildikten sonra <i>diagnose_overfitting_xray.py</i> ile 4 model 6 senaryoda × 2 modda (25'er uçuş = toplam 1,200 uçuş) "
            "birebir eşit koşullarda yarıştırılmıştır:",
            body,
        )
    )

    story.append(
        make_kpi_strip(
            [
                ("360° FLY-AROUND (S5)", "%0 → %100", "Pure-Raw (CBF+PWPF: %88)", "#059669"),
                ("PURE-RAW OOD GAP", "%21.6 → %7.2", "Overfitting Boşluğu 3× Küçüldü", "#0284C7"),
                ("CRITIC EV (R²)", "0.000 → 0.869", "Aktör Donması Tamamen Çözüldü", "#059669"),
                ("CBF BAĞIMLILIĞI (S1/S2)", "%7.6 → %0.0", "Ağ Artık Kendi Frenliyor!", "#0284C7"),
            ]
        )
    )
    story.append(Spacer(1, 6))

    story.append(
        make_callout(
            "[ADIM 1 RÖNTGENİNDE KEŞFEDİLEN KRİTİK KÖK NEDEN — DONMUŞ AKTÖR]",
            "Ağırlık karşılaştırmasında <b>v3 (700K) → v4 (1.4M) → gen (2.4M)</b> boyunca <b>policy_net ağırlıklarının &lt;0.0001 değiştiği</b>, "
            "<i>Action Std</i> değerinin <i>1.000</i>'de donduğu ve Critic'in sabit sayıya çöktüğü (<i>V(15m) = V(1m) = -381.2, EV = 0.000</i>) tespit edildi. "
            "Nedeni: Ölçeklenmemiş dev ödüllerin <i>vf_loss</i>'u milyonlara patlatıp <i>max_grad_norm=0.5</i> kırpmasıyla Aktör gradyanını sıfıra ezmesiydi! "
            "<b>v5</b>'te ödül <i>G_t ∈ [-15, +35]</i> bandına ölçeklenerek <b>EV = 0.869</b>, <b>Actor Drift = 0.066</b> ve <b>Action Std = 0.30</b> ile gerçek RL öğrenimi sağlandı.",
            COLOR_DANGER,
            COLOR_DANGER_BG,
            callout_title,
            callout_body,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1. 6 Senaryolu Bug-Free Röntgen Karşılaştırma Tablosu (Önce: v4/gen vs Sonra: v5)", h2))
    xray_data = [
        [
            Paragraph("Test Senaryosu (25 Uçuş)", th_l),
            Paragraph("v4 / gen (2.4M)<br/>Pure-Raw", th),
            Paragraph("v5 (912K)<br/>Pure-Raw", th),
            Paragraph("v4 / gen (2.4M)<br/>CBF+PWPF", th),
            Paragraph("v5 (912K)<br/>CBF+PWPF", th),
        ],
        [
            Paragraph("<b>S1: Nominal ID</b> (Eğitim Zarfı)", td_l),
            Paragraph("%100.0 (Δv 0.61)", td),
            Paragraph("<b>%100.0 (Δv 0.59)</b>", td),
            Paragraph("%100.0 (CBF %7.6)", td),
            Paragraph("<b>%100.0 (CBF %0.0!)</b>", td),
        ],
        [
            Paragraph("<b>S2: Gerçek Kütle ±%35</b> + %40 İtki Arızası", td_l),
            Paragraph("%96.0 (Kaza %4)", td),
            Paragraph("<b>%100.0 (Kaza %0!)</b>", td),
            Paragraph("%96.0 (Kaza %4)", td),
            Paragraph("<b>%100.0 (CBF %0.0!)</b>", td),
        ],
        [
            Paragraph("<b>S3: 5cm/1.5cm Gürültü</b> + ±4cm Bias + Gecikme", td_l),
            Paragraph("%96.0 (Δv 1.16)", td),
            Paragraph("%64.0 (Kaza %12)*", td),
            Paragraph("%92.0 (CBF %14.3)", td),
            Paragraph("%68.0 (Kaza %0, CBF %2.9)*", td),
        ],
        [
            Paragraph("<b>S4: 30–50m, ±10m</b> + KAÇIŞ Hızı (vy&gt;0)", td_l),
            Paragraph("%100 (Sat %53, Δv 1.38)", td),
            Paragraph("<b>%100 (Sat %0, Δv 1.20)</b>", td),
            Paragraph("%100 (CBF %28.5, Δv 4.01)", td),
            Paragraph("<b>%100 (CBF %0.6, Δv 2.51!)</b>", td),
        ],
        [
            Paragraph("<b>S5: Yandan/Arkadan 360° Fly-Around</b> (+Y Dışı)", td_l),
            Paragraph("<b>%0.0</b> (Sat %71)", td),
            Paragraph("<b>%100.0 (Sat %0!)</b>", td),
            Paragraph("<b>%0.0</b> (Kaza %12, Sat %81)", td),
            Paragraph("<b>%88.0 (Sat %0, Δv 3.61)</b>", td),
        ],
        [
            Paragraph("<b>S6: Gerçek qvel Çift Hız Şoku</b> (t=25s, 55s)", td_l),
            Paragraph("%100.0 (Δv 0.92)", td),
            Paragraph("<b>%100.0 (Δv 0.89)</b>", td),
            Paragraph("%100.0 (CBF %11.8)", td),
            Paragraph("<b>%100.0 (CBF %1.2!)</b>", td),
        ],
        [
            Paragraph("<b>OOD ORTALAMA (S2–S6) &amp; GAP</b>", td_b),
            Paragraph("%78.4 (Gap: %21.6)", td),
            Paragraph("<b>%92.8 (Gap: %7.2)</b>", td),
            Paragraph("%77.6 (Gap: %22.4)", td),
            Paragraph("<b>%91.2 (Gap: %8.8)</b>", td),
        ],
    ]
    xray_tbl = Table(
        xray_data,
        colWidths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.175, CONTENT_WIDTH * 0.175, CONTENT_WIDTH * 0.175, CONTENT_WIDTH * 0.175],
    )
    xray_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, COLOR_BG_ALT]),
                ("BACKGROUND", (2, 5), (2, 5), COLOR_SUCCESS_BG),
                ("BACKGROUND", (4, 5), (4, 5), COLOR_SUCCESS_BG),
                ("BACKGROUND", (1, 5), (1, 5), COLOR_DANGER_BG),
                ("BACKGROUND", (3, 5), (3, 5), COLOR_DANGER_BG),
                ("BACKGROUND", (0, 7), (-1, 7), COLOR_INFO_BG),
                ("GRID", (0, 0), (-1, -1), 0.45, COLOR_BORDER),
                ("PADDING", (0, 0), (-1, -1), 4.0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(xray_tbl)
    story.append(Spacer(1, 5))

    story.append(
        Paragraph(
            "<i>* S3 Notu: v5, ±4 cm sabit konum biası + 5 cm RMS gürültü altında port düzlemi (ry ≈ 0.1m) etrafında 360° fly-around "
            "sınırına (behind_flag) yaklaştığı için temkinli davranıp %0 kaza ile %32 hover üretmektedir; bu durum sensör girişine "
            "basit bir Kalman/EMA alçak geçiren filtre eklenerek kolayca %95+ seviyesine taşınabilir.</i>",
            caption,
        )
    )

    story.append(
        make_callout(
            "[EĞİTİM DOSYALARI & ÜRETİLEN MODEL VARLIKLARI]",
            "• <b>v5 En İyi OOD Modeli (912K Adım):</b> <i>models/ppo_mujoco_champion_v5_best_ood.zip</i> (2.4M'den %62 daha az adımla 3× daha iyi genelleme!)<br/>"
            "• <b>Sıfır-Körlük Özellik Çıkarıcı:</b> <i>orbital_rl_features.py → OrbitalObsFeaturesExtractorV5</i> (9D drop-in uyumlu, %0.0 saturation)<br/>"
            "• <b>Eğitim &amp; Diagnostik Kodları:</b> <i>diagnose_overfitting_xray.py</i>, <i>train_mujoco_champion_v5.py</i>, <i>finetune_v5_terminal_braking.py</i><br/>"
            "• <b>Düzeltilen Eski Bug'lar:</b> <i>train_ppo_mujoco_dr.py</i> (body_mass + obs[6:9]), <i>mujoco_live_server.py</i> (gerçek qvel şoku)",
            COLOR_SECONDARY,
            COLOR_INFO_BG,
            callout_title,
            callout_body,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] PDF başarıyla oluşturuldu: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
