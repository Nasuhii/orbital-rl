#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital-RL: Junior / Anlaşılır Dilde Nihai Proje Kapanış Raporu PDF Üreticisi
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

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(DIR_PATH, "Orbital_RL_Proje_Belgesi_JR.pdf")

FONT_DIR = "/usr/share/fonts/dejavu"
for candidate in ["/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF"]:
    if os.path.exists(os.path.join(candidate, "DejaVuSans.ttf")):
        FONT_DIR = candidate
        break

pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))

MONO_PATH = os.path.join(FONT_DIR, "DejaVuSansMono.ttf")
if os.path.exists(MONO_PATH):
    pdfmetrics.registerFont(TTFont("DejaVuMono", MONO_PATH))
    CODE_FONT = "DejaVuMono"
else:
    CODE_FONT = "DejaVu"

COLOR_PRIMARY = colors.HexColor("#0F172A")
COLOR_SECONDARY = colors.HexColor("#0284C7")
COLOR_SUCCESS = colors.HexColor("#059669")
COLOR_SUCCESS_BG = colors.HexColor("#ECFDF5")
COLOR_WARNING = colors.HexColor("#D97706")
COLOR_WARNING_BG = colors.HexColor("#FFFBEB")
COLOR_INFO_BG = colors.HexColor("#F0F9FF")
COLOR_BG_ALT = colors.HexColor("#F1F5F9")
COLOR_BORDER = colors.HexColor("#CBD5E1")
COLOR_TEXT = colors.HexColor("#1E293B")
COLOR_MUTED = colors.HexColor("#64748B")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 36
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


class NumberedCanvas(canvas.Canvas):
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
        self.setFont("DejaVu", 7.8)
        self.setFillColor(COLOR_MUTED)
        if self._pageNumber > 1:
            self.drawString(MARGIN, PAGE_HEIGHT - 24, "Orbital-RL // Proje Kapanış Belgesi")
            self.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 24, "github.com/Nasuhii/orbital-rl")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(MARGIN, PAGE_HEIGHT - 29, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 29)
        self.line(MARGIN, 28, PAGE_WIDTH - MARGIN, 28)
        self.drawString(MARGIN, 16, "Hazırlayan: Nasuhii | Leo 400 km Otonom Kenetlenme (PPO + MuJoCo)")
        self.drawRightString(PAGE_WIDTH - MARGIN, 16, f"Sayfa {self._pageNumber} / {page_count}")
        self.restoreState()


STYLES = {
    "cover_title": ParagraphStyle("cover_title", fontName="DejaVu-Bold", fontSize=22, leading=28, textColor=COLOR_PRIMARY, spaceAfter=6),
    "cover_sub": ParagraphStyle("cover_sub", fontName="DejaVu", fontSize=11, leading=16, textColor=COLOR_MUTED, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=14, leading=19, textColor=COLOR_PRIMARY, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=COLOR_SECONDARY, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.6, leading=14.5, textColor=COLOR_TEXT, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.4, leading=14, leftIndent=12, textColor=COLOR_TEXT, spaceAfter=4),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Oblique", fontSize=8.3, leading=11.5, alignment=1, textColor=COLOR_MUTED, spaceBefore=3, spaceAfter=10),
    "callout": ParagraphStyle("callout", fontName="DejaVu", fontSize=9.2, leading=13.5, textColor=COLOR_TEXT),
    "code": ParagraphStyle("code", fontName=CODE_FONT, fontSize=8.4, leading=12, textColor=colors.HexColor("#F8FAFC")),
    "th": ParagraphStyle("th", fontName="DejaVu-Bold", fontSize=8.5, leading=11, alignment=1, textColor=colors.white),
    "td": ParagraphStyle("td", fontName="DejaVu", fontSize=8.4, leading=11.5, alignment=1, textColor=COLOR_TEXT),
    "td_left": ParagraphStyle("td_left", fontName="DejaVu", fontSize=8.4, leading=11.5, alignment=0, textColor=COLOR_TEXT),
}


def make_callout(title, text, bg=COLOR_INFO_BG, border=COLOR_SECONDARY):
    p = Paragraph(f"<b>{title}</b><br/>{text}", STYLES["callout"])
    t = Table([[p]], colWidths=[CONTENT_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, border),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def add_img(story, filename, caption, max_h=240):
    path = os.path.join(DIR_PATH, filename)
    if not os.path.exists(path):
        return
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = min(CONTENT_WIDTH / float(w), max_h / float(h))
    img = Image(path, width=w * ratio, height=h * ratio)
    story.append(KeepTogether([img, Paragraph(caption, STYLES["caption"])]))


def make_table(headers, rows, col_widths):
    data = [[Paragraph(h, STYLES["th"]) for h in headers]]
    for row in rows:
        data.append([
            Paragraph(str(cell), STYLES["td_left"] if idx == 0 else STYLES["td"])
            for idx, cell in enumerate(row)
        ])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.4, COLOR_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_pdf():
    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=38,
        title="Orbital-RL Proje Belgesi",
        author="Nasuhii",
    )
    story = []

    # Kapak / Başlık
    story.append(Paragraph("Orbital-RL: Otonom Uzay Aracı Kenetlenme Projesi", STYLES["cover_title"]))
    story.append(Paragraph(
        "Proje Kapanış Belgesi — Fizik Simülasyonu, Karşılaşılan Hatalar, RL Eğitimi ve Test Sonuçları",
        STYLES["cover_sub"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_SECONDARY, spaceAfter=10))

    story.append(make_callout(
        "Proje Özeti (Ne Yaptık?)",
        "Bu projede, Alçak Dünya Yörüngesinde (400 km) 500 kg kütleli bir uzay aracının ±15 N iticiler kullanarak "
        "hedef istasyona çarpmadan kendi kendine kenetlenmesini (docking) sağlayan bir Pekiştirmeli Öğrenme (RL) "
        "sistemi geliştirdik. Başta karşılaştığımız 'liman önünde durup kalma' ve 'yeni açılarda ezberleme' "
        "sorunlarını adım adım çözerek <b>Champion-PPO v5</b> modelinde %100 nominal başarı ve %92.0 dağılım dışı (OOD) "
        "başarıya ulaştık.",
        bg=COLOR_INFO_BG,
        border=COLOR_SECONDARY,
    ))
    story.append(Spacer(1, 10))

    add_img(story, "orbital_rl_executive_presentation_showcase.png",
            "Şekil 1: Orbital-RL genel proje özeti ve simülasyon panosu.", max_h=235)

    # Bölüm 1
    story.append(Paragraph("1. Problem ve Başarı Kriterlerimiz", STYLES["h1"]))
    story.append(Paragraph(
        "Uzayda iki aracın kenetlenmesi dışarıdan basit gibi görünse de sürtünme olmadığı için araç hızlandığında "
        "kendiliğinden yavaşlamıyor. Elimizde sadece ±15 Newton itki üretebilen küçük iticiler var; yani bir araba "
        "gibi son anda frene basamıyoruz, çok önceden ters yöne itki verip yavaşlamamız gerekiyor.",
        STYLES["body"],
    ))
    story.append(make_table(
        ["Parametre", "Değer", "Ne Anlama Geliyor?"],
        [
            ["Yörünge (LEO)", "400 km", "ISS yörüngesi (Clohessy-Wiltshire denklemleri)"],
            ["Araç Kütlesi & İtki", "500 kg / ±15 N", "Maksimum ivme 0.03 m/s² (fren mesafesi uzun)"],
            ["Kenetlenme Yarıçapı", "≤ 0.14 m (~15 cm)", "Aracın burun probunun hedef porta girmesi gereken alan"],
            ["Maks. Temas Hızı", "≤ 0.06 m/s (6 cm/s)", "Bu hızın üstündeki her temas kaza (Crash) sayılır"],
        ],
        [135, 110, CONTENT_WIDTH - 245],
    ))
    story.append(Spacer(1, 8))

    add_img(story, "mission_profile_architecture.png",
            "Şekil 2: Sistemin genel çalışma şeması (Fizik motoru, RL ajanı, güvenlik kalkanı ve eyleyici).", max_h=215)

    story.append(PageBreak())

    # Bölüm 2
    story.append(Paragraph("2. Süreç Boyunca Karşılaştığımız Sorunlar ve Çözümler", STYLES["h1"]))
    story.append(Paragraph(
        "İlk yazdığımız model tek seferde çalışmadı; her aşamada yeni bir problem görüp üzerine giderek modeli geliştirdik:",
        STYLES["body"],
    ))

    story.append(Paragraph("2.1. Hovering (Liman Önünde Durup Kalma) Sorunu", STYLES["h2"]))
    story.append(Paragraph(
        "İlk modelimize (Naive PPO) basitçe 'yaklaşırsan artı puan, 6 cm/s üstünde çarparsan -100 ceza' verdik. "
        "Model de -100 ceza almamak için limana 2–3 metre kala tamamen durdu ve süre bitene kadar bekledi (%0 başarı, %100 hovering). "
        "Bunu çözmek için mesafeye göre izin verilen hızı düşüren bir hız sınırı eğrisi (tanh) ve yerinde durunca "
        "her saniye küçük ceza kesen PBRS ödül yapısı ekledik.",
        STYLES["body"],
    ))

    story.append(Paragraph("2.2. MuJoCo Fiziği, Güvenlik Kalkanı (CBF) ve PWPF İticiler", STYLES["h2"]))
    story.append(Paragraph(
        "Çarpışma ve temasları gerçek dünyaya daha yakın test edebilmek için Google DeepMind MuJoCo 3D fizik "
        "motoruna geçtik. Ancak burada hiçbir emniyet önlemi olmadan eğitilen modellerin yarısından fazlası "
        "(%56.7) limana çok hızlı girip kaza yaptı. Bunu çözmek için sisteme iki temel parça ekledik: "
        "<b>(1) Güvenlik Kalkanı (CBF-QP):</b> Yapay zeka limana tehlikeli bir hızla yaklaştığında araya girip "
        "otomatik fren yaptıran bir emniyet kilidi. <b>(2) Aç/Kapa İtici Dönüştürücü (PWPF):</b> Gerçek uzay "
        "aracı iticileri arabadaki gaz pedalı gibi ara değerler (örneğin 3.5 N) üretemez; ya tamamen kapalıdır "
        "(0 N) ya da tam güç açıktır (15 N). Bu yüzden modelin istediği kuvveti kısa aç/kapa darbelerine çevirdik. "
        "Bu güvenli kontrol yapısıyla (Integrated Safe-GNC) yaptığımız 30 farklı rastgele uçuş testinin "
        "tamamında (30/30 — %100 başarı) aracı kazasız şekilde kenetledik.",
        STYLES["body"],
    ))

    add_img(story, "phase_plane_safety_corridors.png",
            "Şekil 3: Faz düzleminde (hız-mesafe) güvenli yaklaşma koridoru ve CBF kalkanının sınırı.", max_h=205)

    add_img(story, "rl_ai_training_and_benchmark_dashboard.png",
            "Şekil 4: 30 uçuşluk Monte Carlo testinde modellerin kıyaslama grafikleri.", max_h=215)

    story.append(PageBreak())

    # Bölüm 3
    story.append(Paragraph("3. Ezberleme (Overfitting) Kontrolü ve Champion-PPO v5", STYLES["h1"]))
    story.append(Paragraph(
        "İlk şampiyon modellerimiz (v3 ve v4) normal testte %100 başarı veriyordu ama gerçekten öğrenip öğrenmediğini "
        "görmek için zorlu OOD (eğitimde hiç görmediği durumlar) testi yazdık (diagnose_overfitting_xray.py). "
        "Test edince modelin sadece tam önden (+Y ekseninden) gelmeyi ezberlediğini, yandan veya arkadan başlatınca "
        "başarının %0'a düştüğünü gördük. Ayrıca kodda 4 önemli hata yakaladık:",
        STYLES["body"],
    ))
    for item in [
        "<b>1. Aktör Donması:</b> Ödül sayıları çok büyük olduğu için Critic kaybı patlayıp sinir ağının öğrenmesini kilitliyormuş.",
        "<b>2. Kütle DR Hatası:</b> MuJoCo'da kütle değiştirirken <i>mj_setConst</i> çağrılmadığı için fiziksel kütle sabit kalıyormuş.",
        "<b>3. Veri Sızıntısı:</b> Sensör gürültüsü ekledikten sonra mesafe/hız normu temiz veriden hesaplandığı için model kopya çekiyormuş.",
        "<b>4. Şok Hatası:</b> Canlı hız şoku doğrudan fizik motorunun hız vektörüne (<i>qvel</i>) yazılmıyormuş.",
    ]:
        story.append(Paragraph("• " + item, STYLES["bullet"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Bu 4 hatayı düzeltip ödülleri küçülttük (Critic R² skoru 0.000'dan 0.869'a çıktı), uzak mesafe körlüğünü "
        "kaldırmak için asinh + LayerNorm ekledik ve aracı 360 derece farklı açılardan, kütle/itici arızalarıyla "
        "eğiterek <b>Champion-PPO v5</b> modelini çıkardık.",
        STYLES["body"],
    ))

    story.append(Paragraph("3.1. 30 Uçuşluk Nominal Monte Carlo Sonuçları", STYLES["h2"]))
    story.append(make_table(
        ["Model", "Başarı", "Hover", "Kaza", "Son Hız", "Harcanan Δv"],
        [
            ["Naive PPO", "%0.0", "%100.0", "%0.0", "0.835 m/s", "4.06 m/s"],
            ["PBRS PPO", "%0.0", "%100.0", "%0.0", "0.145 m/s", "1.73 m/s"],
            ["MuJoCo DR PPO (Kalkansız)", "%0.0", "%43.3", "%56.7", "1.409 m/s", "2.68 m/s"],
            ["Integrated Safe-GNC", "%100.0 (30/30)", "%0.0", "%0.0", "0.0441 m/s", "0.94 m/s"],
        ],
        [150, 78, 55, 55, 85, CONTENT_WIDTH - 423],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.2. Champion-PPO v5 Zorlu OOD Testleri (Her Senaryo 25 Uçuş)", STYLES["h2"]))
    story.append(make_table(
        ["Test Senaryosu", "v5 Kalkansız (Pure-Raw)", "v5 Kalkanlı (CBF+PWPF)", "Açıklama"],
        [
            ["S1: Normal Başlangıç (ID)", "%100.0", "%100.0", "Ort. Δv = 0.577 m/s"],
            ["S2: ±%35 Kütle + %40 İtici Arızası", "%100.0", "%100.0", "Arızaya rağmen tam başarı"],
            ["S3: Yüksek Gürültü + Bias + Gecikme", "%60.0", "%56.0", "Tek zayıf nokta (gürültü)"],
            ["S4: 30–50m Uzaktan Kaçış (vy > 0)", "%100.0", "%96.0", "Kaçan hedefi yakaladı"],
            ["S5: 360° Yandan / Arkadan Yaklaşma", "%100.0", "%88.0", "Eski modelde %0 idi"],
            ["S6: Uçuş Sırasında Çift Hız Şoku", "%100.0", "%100.0", "Darbeden sonra toparladı"],
            ["OOD Ortalama Başarı", "%92.0 (Gap: %8.0)", "%88.0 (Gap: %12.0)", "Genelleme 3 kat iyileşti"],
        ],
        [170, 110, 110, CONTENT_WIDTH - 390],
    ))
    story.append(Spacer(1, 8))

    story.append(make_callout(
        "Dürüst Not: Eksik Kalan Tek Kısım (S3 Senaryosu)",
        "v5 modeli S1, S2, S4, S5 ve S6'da kalkansız %100 başarı aldı. Sadece S3'te (5 cm konum / 1.5 cm/s hız "
        "gürültüsü + ±4 cm sabit sapma + 0.5 sn gecikme) başarı %60'ta kaldı. Çünkü model şu an sadece son ana "
        "bakıyor; gelecekte son 4–8 adımın geçmişini veren Frame Stacking veya LSTM eklenirse bu da çözülebilir.",
        bg=COLOR_WARNING_BG,
        border=COLOR_WARNING,
    ))

    story.append(PageBreak())

    # Bölüm 4
    story.append(Paragraph("4. Simülasyon Görselleri ve Canlı Kontrol Paneli", STYLES["h1"]))
    story.append(Paragraph(
        "Simülasyondaki kenetlenme sürecini hem adım adım çoklu kamera görüntüleriyle hem de kokpit HUD "
        "göstergeleriyle kaydettik. Ayrıca tarayıcıdan canlı izleyebilmek için 30 FPS web arayüzü kurduk.",
        STYLES["body"],
    ))

    add_img(story, "mujoco_multicam_docking_sequence.png",
            "Şekil 5: 20 metreden kenetlenmeye kadar 5 kritik görev kapısının 3 farklı kameradan görünümü.", max_h=225)

    add_img(story, "optical_nav_sensor_hud.png",
            "Şekil 6: Burun kamerası (FPV) kokpit HUD göstergeleri: hız bandı, mesafe bandı ve itki durumu.", max_h=225)

    # Bölüm 5
    story.append(Paragraph("5. Projeyi Çalıştırma ve Öğrendiklerimiz", STYLES["h1"]))
    code_box = Table(
        [[Paragraph(
            "# Canlı 30 FPS MuJoCo arayüzünü açmak için (http://localhost:8055/):\n"
            "python mujoco_live_server.py 8055\n\n"
            "# Zorlu OOD testlerini tekrar çalıştırmak için:\n"
            "python diagnose_overfitting_xray.py",
            STYLES["code"],
        )]],
        colWidths=[CONTENT_WIDTH],
    )
    code_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(code_box)
    story.append(Spacer(1, 8))

    story.append(make_callout(
        "Bu Projeden Çıkardığımız 3 Temel Ders",
        "1) Eğitimde %100 almak modelin gerçekten öğrendiği anlamına gelmiyor; mutlaka görmediği açılarda test etmek lazım.<br/>"
        "2) Ödül ölçeği ve veri sızıntısı en sinsi hatalar; küçük bir gürültü sızıntısı bile tüm testi yanıltabiliyor.<br/>"
        "3) Yapay zekanın önüne matematiksel bir güvenlik kalkanı (CBF) koymak, gerçekçi sistemler için en sağlıklı yol.",
        bg=COLOR_SUCCESS_BG,
        border=COLOR_SUCCESS,
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF oluşturuldu: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
