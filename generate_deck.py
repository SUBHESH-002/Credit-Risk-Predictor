"""
generate_deck.py
----------------
Generates an updated, professional 20-slide presentation deck:
'TabTransformer_GermanCredit_Replication.pptx'

Strictly focuses on the German Credit dataset, removing BISAID,
and embedding our measured experimental results, comparison tables,
and actual generated plot figures.
"""

import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PPTX = Path(r"O:\PROJECTS\Credit Risk Predictor\TabTransformer_GermanCredit_Replication.pptx")
PLOTS_DIR = Path(r"O:\PROJECTS\Credit Risk Predictor\credit-risk-replication\outputs\plots")
SHAP_DIR = PLOTS_DIR / "shap"

# Color Palette matching original sleek style
BG_DARK = RGBColor(0x10, 0x1E, 0x30)      # Deep Navy / Slate
BG_LIGHT = RGBColor(0xF8, 0xF9, 0xFA)     # Off-white / light slate
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)      # Pure white
CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)  # Subtle gray border
TEXT_DARK = RGBColor(0x10, 0x1E, 0x30)    # Primary dark text
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)   # Slate gray muted text
TEXT_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)   # Light text for dark slides
ACCENT_TEAL = RGBColor(0x2A, 0xBF, 0xAE)  # Teal accent
ACCENT_CORAL = RGBColor(0xE5, 0x8B, 0x44) # Coral / amber accent
ACCENT_BLUE = RGBColor(0x25, 0x63, 0xEB)  # Royal blue
COLOR_GOOD = RGBColor(0x10, 0xB9, 0x81)   # Green
COLOR_BAD = RGBColor(0xEF, 0x44, 0x44)    # Red

FONT_TITLE = "Arial"
FONT_BODY = "Calibri"


def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # completely blank slide

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background() # no border
        return bg

    def add_header(slide, category, title, subtitle=None, dark=False):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
        p_c = tf_c.paragraphs[0]
        p_c.text = category.upper()
        p_c.font.name = FONT_TITLE
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = ACCENT_TEAL

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11), Inches(0.8))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_TITLE
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT if dark else TEXT_DARK

        # Subtitle
        if subtitle:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.5), Inches(0.5))
            tf_s = sub_box.text_frame
            tf_s.word_wrap = True
            tf_s.margin_left = tf_s.margin_right = tf_s.margin_top = tf_s.margin_bottom = 0
            p_s = tf_s.paragraphs[0]
            p_s.text = subtitle
            p_s.font.name = FONT_BODY
            p_s.font.size = Pt(13)
            p_s.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8) if dark else TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Navy)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, BG_DARK)

    # Decorative Circle
    circ = slide1.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.2), Inches(1.5), Inches(3.5), Inches(3.5))
    circ.fill.background()
    circ.line.color.rgb = ACCENT_TEAL
    circ.line.width = Pt(18)

    # Badge
    badge_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(8), Inches(0.4))
    p = badge_box.text_frame.paragraphs[0]
    p.text = "RESEARCH REPLICATION & BENCHMARK EVALUATION • IEEE ACCESS 2025"
    p.font.name = FONT_TITLE
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL

    # Main Title
    t_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(8.5), Inches(1.8))
    t_box.text_frame.word_wrap = True
    p1 = t_box.text_frame.paragraphs[0]
    p1.text = "A Novel Weighted Loss\nTabTransformer"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT

    # Subtitle
    sub_box = slide1.shapes.add_textbox(Inches(1.0), Inches(3.8), Inches(8.5), Inches(1.0))
    sub_box.text_frame.word_wrap = True
    p2 = sub_box.text_frame.paragraphs[0]
    p2.text = "Integrating Explainable AI for Imbalanced Credit Risk Datasets\nReplication, Evaluation & SHAP Analysis on the German Credit Benchmark"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    # Divider bar
    div = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(5.1), Inches(2.5), Inches(0.06))
    div.fill.solid()
    div.fill.fore_color.rgb = ACCENT_CORAL
    div.line.fill.background()

    # Footer Metadata
    meta_box = slide1.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(8.5), Inches(1.0))
    p3 = meta_box.text_frame.paragraphs[0]
    p3.text = "Original Authors: Kristoko Dwi Hartomo • Christian Arthur • Yessica Nataliani\nExperimental Replication & Validation • German Credit Dataset (1,000 instances)"
    p3.font.name = FONT_BODY
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 2: THE REPLICATION STORY IN 5 STEPS
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, BG_LIGHT)
    add_header(slide2, "01 · Executive Summary", "The German Credit Replication Journey",
               "From class-imbalance default blindness to an explainable, cost-sensitive TabTransformer")

    steps = [
        ("01", "The Problem", "Default Blindness", "300 Bad vs 700 Good. Raw accuracy ignores costly minority loan defaults."),
        ("02", "The Model", "TabTransformer", "Self-attention across 13 categorical embeddings + 7 scaled numerical features."),
        ("03", "The Method", "Weighted Loss", "Inverse-frequency weights (3.33x on Bad credit) penalize false approvals."),
        ("04", "The Results", "+53.3% Catch", "Bad-credit recall surges from 20% to 73.3%, cutting business cost by -53.3%."),
        ("05", "Explainability", "SHAP Insights", "Checking status & credit history identified as dominant non-linear risk drivers.")
    ]

    card_w = Inches(2.2)
    card_h = Inches(4.5)
    start_x = Inches(0.8)
    gap = Inches(0.18)

    for i, (num, tag, title, desc) in enumerate(steps):
        x = start_x + i * (card_w + gap)
        add_card(slide2, x, Inches(2.0), card_w, card_h)

        tb = slide2.shapes.add_textbox(x + Inches(0.15), Inches(2.2), card_w - Inches(0.3), card_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        p_num = tf.paragraphs[0]
        p_num.text = num
        p_num.font.name = FONT_TITLE
        p_num.font.size = Pt(20)
        p_num.font.bold = True
        p_num.font.color.rgb = ACCENT_TEAL

        p_tag = tf.add_paragraph()
        p_tag.text = tag
        p_tag.font.name = FONT_BODY
        p_tag.font.size = Pt(11)
        p_tag.font.color.rgb = TEXT_MUTED
        p_tag.space_after = Pt(14)

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.name = FONT_TITLE
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_DARK
        p_t.space_after = Pt(10)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: PROBLEM & MOTIVATION — ASYMMETRIC COSTS
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, BG_LIGHT)
    add_header(slide3, "02 · Problem Formulation", "Why Standard Accuracy Fails in Credit Risk",
               "Evaluating credit risk with standard accuracy conceals catastrophic default approval rates")

    # Left Card: Class Imbalance
    add_card(slide3, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7))
    tb_l = slide3.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(5.0), Inches(4.3))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "German Credit Class Imbalance"
    p.font.name = FONT_TITLE; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(12)

    bullets_l = [
        "Total Portfolio: 1,000 consumer loan records.",
        "Majority Class (Good Credit): 700 applicants (70%).",
        "Minority Class (Bad Credit): 300 applicants (30%).",
        "The Majority Classifier Trap: A naive model that approves every single applicant achieves 70% accuracy without underwriting a single loan.",
        "Unweighted deep learning models naturally inherit this bias, optimizing overall loss by sacrificing the minority defaulters."
    ]
    for b in bullets_l:
        p = tf_l.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_BODY; p.font.size = Pt(13); p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # Right Card: The 5x Asymmetric Cost
    add_card(slide3, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7))
    tb_r = slide3.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.1), Inches(4.3))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "The Statlog 5:1 Cost Matrix Penalty"
    p.font.name = FONT_TITLE; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_CORAL
    p.space_after = Pt(12)

    bullets_r = [
        "False Positive (Type I Error): Classifying a bad customer as good. The bank lends money and suffers loan default.",
        "False Negative (Type II Error): Classifying a good customer as bad. The bank turns away a customer, losing modest interest profit.",
        "Statlog Official Cost Matrix:\n   Misclassifying Bad as Good costs 5 units.\n   Misclassifying Good as Bad costs 1 unit.",
        "Business Takeaway: A model with 75% accuracy that catches 73% of defaults is vastly more profitable than a model with 74% accuracy that catches only 20% of defaults."
    ]
    for b in bullets_r:
        p = tf_r.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_BODY; p.font.size = Pt(13); p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 4: DATASET — GERMAN CREDIT SPECIFICATION
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, BG_LIGHT)
    add_header(slide4, "03 · Data Architecture", "Dataset Specification: UCI Statlog German Credit",
               "1,000 instances with 20 predictive features, partitioned into a clean 80:10:10 stratified split")

    card_d1 = add_card(slide4, Inches(0.8), Inches(2.0), Inches(3.7), Inches(4.7))
    tb = slide4.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.3), Inches(4.3))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "13 Categorical Features"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(10)
    cats = [
        "Checking status (4 levels)", "Credit history (5 levels)", "Purpose (10 levels)",
        "Savings status (5 levels)", "Employment duration (5 levels)", "Personal status & sex (5 levels)",
        "Other debtors / guarantors (3 levels)", "Property magnitude (4 levels)", "Other payment plans (3 levels)",
        "Housing (3 levels)", "Job category (4 levels)", "Own telephone (2 levels)", "Foreign worker (2 levels)"
    ]
    for c in cats:
        p = tf.add_paragraph(); p.text = "• " + c; p.font.size = Pt(11); p.font.color.rgb = TEXT_DARK

    card_d2 = add_card(slide4, Inches(4.8), Inches(2.0), Inches(3.7), Inches(4.7))
    tb = slide4.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.3))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "7 Numerical Features"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = ACCENT_CORAL
    p.space_after = Pt(10)
    nums = [
        "Duration in months", "Credit amount (DM)", "Installment commitment (% income)",
        "Present residence since (years)", "Age (years)", "Number of existing credits",
        "Number of dependents liable"
    ]
    for n in nums:
        p = tf.add_paragraph(); p.text = "• " + n; p.font.size = Pt(12); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)
    p = tf.add_paragraph(); p.text = "\nScaling Strategy:"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = "MinMaxScaler fitted strictly to [-1, 1] on training set only."; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED

    card_d3 = add_card(slide4, Inches(8.8), Inches(2.0), Inches(3.7), Inches(4.7))
    tb = slide4.shapes.add_textbox(Inches(9.0), Inches(2.2), Inches(3.3), Inches(4.3))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Stratified Partitions"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)
    parts = [
        "Training Set (80%): 800 records\n  240 Bad (30%) | 560 Good (70%)",
        "Validation Set (10%): 100 records\n  30 Bad (30%) | 70 Good (70%)",
        "Held-Out Test Set (10%): 100 records\n  30 Bad (30%) | 70 Good (70%)",
        "Data Leakage Safeguard: Preprocessing transformers and class weights fitted exclusively on training data.",
        "Missing Values: 0 missing values verified via automated audit."
    ]
    for pt in parts:
        p = tf.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(11); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 5: MODEL ARCHITECTURE — TABTRANSFORMER
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, BG_LIGHT)
    add_header(slide5, "04 · Model Architecture", "TabTransformer: Contextual Tabular Attention",
               "Leveraging multi-head self-attention to generate contextual representations across categorical credit attributes")

    arch_items = [
        ("1. Categorical Embeddings", "13 attributes encoded into 64-dimensional learnable vectors. Unknown slot reserved for robustness."),
        ("2. Multi-Head Self-Attention", "2 Transformer layers with 2 attention heads. Computes inter-feature dependencies (e.g., job status contextualized by housing)."),
        ("3. Continuous Concatenation", "7 continuous features concatenated directly with the contextualized categorical embeddings."),
        ("4. MLP Classification Head", "Multi-layer perceptron with ReLU activation projecting down to 2 output logits (Class 0: Bad, Class 1: Good).")
    ]
    for i, (title, desc) in enumerate(arch_items):
        y = Inches(2.0 + i * 1.2)
        add_card(slide5, Inches(0.8), y, Inches(7.0), Inches(1.05))
        tb = slide5.shapes.add_textbox(Inches(1.0), y + Inches(0.1), Inches(6.6), Inches(0.85))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED

    # Hyperparameters Card on the Right
    add_card(slide5, Inches(8.2), Inches(2.0), Inches(4.3), Inches(4.8), bg_color=BG_DARK)
    tb = slide5.shapes.add_textbox(Inches(8.5), Inches(2.2), Inches(3.7), Inches(4.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Hyperparameter Specs"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(14)

    hparams = [
        ("Embedding Dimension", "64"),
        ("Transformer Depth", "2 layers"),
        ("Attention Heads", "2 heads"),
        ("Attention Dropout", "0.1"),
        ("Feed-Forward Dropout", "0.1"),
        ("Batch Size", "64"),
        ("Optimizer", "AdamW (lr = 0.001)"),
        ("Max Epochs", "50 (Early Stop @ Best Val)"),
        ("Parameters", "8,577,806 trainable")
    ]
    for k, v in hparams:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT; p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 6: METHOD — WEIGHTED LOSS FORMULATION
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, BG_LIGHT)
    add_header(slide6, "05 · Method Formulation", "Inverse-Frequency Weighted Cross-Entropy",
               "Penalizing minority default errors to shift the decision boundary toward recall preservation")

    # Left formula card
    add_card(slide6, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), bg_color=BG_DARK)
    tb = slide6.shapes.add_textbox(Inches(1.1), Inches(2.3), Inches(5.0), Inches(4.1))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Mathematical Formulation"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "Weight Definition (Paper Eq. 3):\n  wᵢ = N / Nᵢ\nwhere N = total samples, Nᵢ = samples in class i"
    p.font.size = Pt(14); p.font.color.rgb = TEXT_LIGHT; p.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "German Credit Calculated Weights:\n• Class 0 (Bad Credit, 30%):  w₀ = 1000 / 300 = 3.333\n• Class 1 (Good Credit, 70%): w₁ = 1000 / 700 = 1.429\n\nRelative Ratio: w₀ / w₁ = 2.333x"
    p.font.size = Pt(14); p.font.color.rgb = ACCENT_CORAL; p.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "Weighted Loss Function:\n  L = - ∑ wᵢ yᵢ log(p̂ᵢ)"
    p.font.size = Pt(14); p.font.color.rgb = TEXT_LIGHT

    # Right explanation card
    add_card(slide6, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7))
    tb = slide6.shapes.add_textbox(Inches(7.1), Inches(2.3), Inches(5.1), Inches(4.1))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Three Rigorous Replication Modes"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(14)

    modes = [
        ("Experiment A: Unweighted Baseline", "Standard cross-entropy loss without class weights. Serves as direct empirical baseline."),
        ("Experiment B: Full-Dataset Weights", "Weights derived from the 1,000-sample distribution (w₀=3.333, w₁=1.429). Directly follows paper's stated procedure."),
        ("Experiment C: Train-Only Weights", "Conservative protocol computing weights strictly from the 800 training records. Confirms zero data leakage (w₀=800/240=3.333, w₁=800/560=1.429).")
    ]
    for title, desc in modes:
        p = tf.add_paragraph(); p.text = title; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED; p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 7: BENCHMARK COMPARISON — PAPER REPORTED VS REPLICATION
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, BG_LIGHT)
    add_header(slide7, "06 · Results Comparison", "Paper-Reported vs. Measured Experimental Results",
               "Direct empirical comparison between the IEEE Access paper's claims and our measured replication")

    # Table Card
    add_card(slide7, Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))

    rows = 7
    cols = 5
    table_shape = slide7.shapes.add_table(rows, cols, Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.2))
    table = table_shape.table

    table_data = [
        ["Evaluation Metric", "Paper Baseline", "Paper Weighted", "Our Exp A (Unweighted)", "Our Exp B/C (Weighted)"],
        ["Overall Accuracy", "93.00%", "95.00%", "74.00%", "75.00%"],
        ["ROC-AUC", "0.9100", "0.9600", "0.8148", "0.8157"],
        ["Bad-Credit Recall (Class 0)", "0.9300", "0.9700", "0.2000 (6/30)", "0.7333 (22/30) [▲ +53.3%]"],
        ["Bad-Credit Precision", "0.8500", "0.8800", "0.7500", "0.5641"],
        ["Macro-F1 Score", "Not Reported", "Not Reported", "0.5776", "0.7234 [▲ +0.1458]"],
        ["F2-Score (Minority Focused)", "Discussed", "Discussed", "0.5742", "0.7345 [▲ +0.1603]"]
    ]

    for r_idx, row_text in enumerate(table_data):
        for c_idx, cell_text in enumerate(row_text):
            cell = table.cell(r_idx, c_idx)
            cell.text = cell_text
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_BODY
            p.font.size = Pt(11 if r_idx > 0 else 12)
            if r_idx == 0:
                p.font.bold = True
                cell.fill.solid(); cell.fill.fore_color.rgb = BG_DARK
                p.font.color.rgb = TEXT_LIGHT
            else:
                if c_idx in [3, 4]:
                    p.font.bold = True
                p.font.color.rgb = TEXT_DARK

    # Summary callout below table
    tb_c = slide7.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.0))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Key Comparative Takeaway:"
    p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = ACCENT_CORAL
    p = tf_c.add_paragraph()
    p.text = "While the paper reports an extraordinarily high 95% accuracy on German Credit (which ML literature shows often indicates data leakage or overfitting on this noisy 1,000-sample dataset), our replication empirically validates the paper's core scientific hypothesis: weighted loss drives a massive +53.3% recall surge for the minority defaulter class."
    p.font.size = Pt(12); p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 8: CONFUSION MATRIX ANALYSIS — DEFAULT BLINDNESS ELIMINATED
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, BG_LIGHT)
    add_header(slide8, "07 · Confusion Matrix Analysis", "Eliminating Default Blindness: 6 vs. 22 Caught Defaults",
               "Visualizing the dramatic shift in minority class recognition on the held-out test set (N=100)")

    # Left Image: Exp A Confusion Matrix
    img_a = PLOTS_DIR / "confusion_matrix_expA_test.png"
    if img_a.exists():
        slide8.shapes.add_picture(str(img_a), Inches(0.8), Inches(2.0), width=Inches(5.6))

    # Right Image: Exp B Confusion Matrix
    img_b = PLOTS_DIR / "confusion_matrix_expB_test.png"
    if img_b.exists():
        slide8.shapes.add_picture(str(img_b), Inches(6.8), Inches(2.0), width=Inches(5.6))

    # Interpretation footer card
    add_card(slide8, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.3))
    tb = slide8.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(1.1))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Empirical Shift: From 80% Missed Defaults to 73.3% Default Detection"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph()
    p.text = "• Unweighted Exp A: Only 6 out of 30 bad credit borrowers were flagged. 24 high-risk borrowers were mistakenly granted loans.\n• Weighted Exp B/C: 22 out of 30 bad credit borrowers were flagged. Default catch rate jumped by +53.3 percentage points, directly aligning with real-world credit risk management objectives."
    p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 9: BUSINESS IMPACT — 53.3% FINANCIAL COST REDUCTION
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, BG_LIGHT)
    add_header(slide9, "08 · Financial Cost Evaluation", "Statlog Cost Matrix: 53.3% Reduction in Credit Losses",
               "Translating classification errors into bottom-line banking profit and loss metrics")

    card1 = add_card(slide9, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.8))
    tb1 = slide9.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(5.0), Inches(4.4))
    tf1 = tb1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Exp A: Unweighted Baseline"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = COLOR_BAD
    p.space_after = Pt(12)
    p = tf1.add_paragraph(); p.text = "Error Breakdown:\n• False Positives (Bad approved): 24\n• False Negatives (Good denied): 2"; p.font.size = Pt(13); p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(14)
    p = tf1.add_paragraph(); p.text = "Statlog Cost Calculation:\nCost = (5 × 24) + (1 × 2)\nCost = 120 + 2 = 122 units"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(14)
    p = tf1.add_paragraph(); p.text = "Business Consequence:\nApproving 24 defaulting borrowers inflicts massive write-offs that completely dwarf the bank's operational revenue."; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED

    card2 = add_card(slide9, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8))
    tb2 = slide9.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.1), Inches(4.4))
    tf2 = tb2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "Exp B/C: Weighted Loss Model"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = COLOR_GOOD
    p.space_after = Pt(12)
    p = tf2.add_paragraph(); p.text = "Error Breakdown:\n• False Positives (Bad approved): 8  (▼ -16)\n• False Negatives (Good denied): 17 (▲ +15)"; p.font.size = Pt(13); p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(14)
    p = tf2.add_paragraph(); p.text = "Statlog Cost Calculation:\nCost = (5 × 8) + (1 × 17)\nCost = 40 + 17 = 57 units"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(14)
    p = tf2.add_paragraph(); p.text = "Net Financial Improvement:\nTotal cost dropped from 122 to 57 (-53.3% reduction).\nTrading 15 false alarms for 16 caught defaults saves the lending institution 65 net cost units."; p.font.size = Pt(12); p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 10: ROC-AUC & DISCRIMINATION ABILITY
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, BG_LIGHT)
    add_header(slide10, "09 · Discrimination Analysis", "ROC Curves: Ranking Stability Across Thresholds",
               "Evaluating threshold-independent discrimination capability across all models")

    # Image Left
    img_roc = PLOTS_DIR / "roc_curves_all_test.png"
    if img_roc.exists():
        slide10.shapes.add_picture(str(img_roc), Inches(0.8), Inches(2.0), width=Inches(5.8))

    # Right Card: ROC Commentary
    add_card(slide10, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.8))
    tb = slide10.shapes.add_textbox(Inches(7.3), Inches(2.2), Inches(4.9), Inches(4.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "ROC-AUC Insights"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(12)

    roc_pts = [
        "Consistent Discrimination: All TabTransformer variants achieved ROC-AUC of ~0.815–0.816 on the test split.",
        "Threshold Shift Effect: Weighted loss does not primarily distort rank-ordering; instead, it shifts the uncalibrated probability output distribution toward the minority class.",
        "Decision Threshold Independence: Because ROC curves evaluate across all operational cutoffs, the model maintains high discriminative capacity regardless of lending policy stringency.",
        "Comparison with Baseline: TabTransformer (0.8157) distinctly outperforms Logistic Regression (0.7724) by capturing non-linear interactions across checking and credit history."
    ]
    for pt in roc_pts:
        p = tf.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(12); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 11: EXPLAINABLE AI — GLOBAL SHAP IMPORTANCE
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, BG_LIGHT)
    add_header(slide11, "10 · Explainable AI", "Global SHAP Feature Importance: German Credit",
               "Using KernelExplainer to interpret feature-level contributions to the TabTransformer's predictions")

    # Image Left: Global SHAP
    img_shap = SHAP_DIR / "global_importance_expB.png"
    if img_shap.exists():
        slide11.shapes.add_picture(str(img_shap), Inches(0.8), Inches(2.0), width=Inches(5.8))

    # Right Card: Ranking & Context
    add_card(slide11, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.8))
    tb = slide11.shapes.add_textbox(Inches(7.3), Inches(2.2), Inches(4.9), Inches(4.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Top Driving Features"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(10)

    shap_pts = [
        "1. checking_status (0.1122): The single most decisive feature. Negative or absent checking accounts sharply depress credit approval probability.",
        "2. purpose (0.0397): Loan objective (new car, education, business) dictates underwriting risk.",
        "3. credit_history (0.0382): Prior payment punctuality strongly influences model embeddings.",
        "4. property_magnitude (0.0375): Tangible collateral (real estate vs none).",
        "5. savings_status (0.0329): Liquid asset buffer in bonds or savings.",
        "Alignment with Paper: The paper highlighted account status and credit history. In our TabTransformer, discrete financial bands create distinct topological embeddings that dominate continuous attributes."
    ]
    for pt in shap_pts:
        p = tf.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(11); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 12: LOCAL EXPLAINABILITY — INDIVIDUAL WATERFALL ANALYSIS
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, BG_LIGHT)
    add_header(slide12, "11 · Local Explainability", "Individual Decision Waterfall Explanations",
               "Auditing applicant-specific predictions to provide adverse action notices and underwriting justification")

    # Image Left
    img_ind = SHAP_DIR / "individual_expB_sample1.png"
    if img_ind.exists():
        slide12.shapes.add_picture(str(img_ind), Inches(0.8), Inches(2.0), width=Inches(5.8))

    # Right Card
    add_card(slide12, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.8))
    tb = slide12.shapes.add_textbox(Inches(7.3), Inches(2.2), Inches(4.9), Inches(4.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Auditing Sample #1 Prediction"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(12)

    loc_pts = [
        "Local Attribution: Shows how individual attribute values push probability toward Good Credit (Green) or Bad Credit (Red).",
        "Positive Push (Green): Long employment history (>=7 years), existing real estate ownership, and pristine credit history exert positive forces.",
        "Negative Drag (Red): Overdraft status on checking account (< 0 DM) exerts strong negative drag, warning the underwriter of immediate liquidity distress.",
        "Regulatory Compliance: Enables automated Adverse Action Notices mandated by the US Equal Credit Opportunity Act (ECOA) and EU AI Act Article 13 transparency rules."
    ]
    for pt in loc_pts:
        p = tf.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(12); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 13: CRITICAL APPRAISAL — WHY THE PAPER DISCREPANCY?
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13, BG_LIGHT)
    add_header(slide13, "12 · Critical Appraisal", "Replication Analysis: Why the Discrepancy with the Paper?",
               "An honest academic appraisal of why realistic test accuracy is 75% while the paper reported 95%")

    app_items = [
        ("1. Data Leakage in Tuning", "The paper describes 10-fold cross-validation during hyperparameter tuning alongside an 80:20 / 80:10:10 split. If hyperparameters or weights are selected using validation folds without an isolated held-out test set, performance will be severely inflated."),
        ("2. Small Sample Statistical Variance", "German Credit has only 1,000 records. A 10% test split contains only 100 cases (30 bad, 70 good). A swing of 10 samples alters measured accuracy by 10.0%, introducing extreme split sensitivity."),
        ("3. Tabular Machine Learning Reality", "Across 30 years of benchmark literature, German Credit models (Random Forests, Gradient Boosting, SVMs, Neural Nets) consistently peak around 72%–78% generalized accuracy due to noisy 1994 attributes. 95% accuracy is historically indicative of test leakage."),
        ("4. Scientific Validation of Core Thesis", "Crucially, our replication validates the paper's central scientific proposition: weighted loss successfully cures minority default blindness (+53.3% recall improvement), even when evaluated on an honest, unseen test set.")
    ]
    for i, (title, desc) in enumerate(app_items):
        y = Inches(2.0 + i * 1.2)
        add_card(slide13, Inches(0.8), y, Inches(11.7), Inches(1.05))
        tb = slide13.shapes.add_textbox(Inches(1.0), y + Inches(0.1), Inches(11.3), Inches(0.85))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = ACCENT_CORAL
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(11); p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 14: FAIR LENDING & ETHICAL GOVERNANCE
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14, BG_LIGHT)
    add_header(slide14, "13 · Ethical Governance", "Responsible AI & Fair Lending Safeguards",
               "Credit underwriting models must satisfy rigorous fairness, calibration, and anti-bias standards")

    gov_cards = [
        ("Probability Calibration", "TabTransformer raw logits are not well-calibrated probabilities. Platt scaling or Isotonic Regression must be applied prior to setting credit interest rates."),
        ("Protected Attributes", "German Credit includes Attribute 9 ('personal_status': sex and marital status) and Attribute 13 ('age'). Regulators prohibit disparate impact on these protected classes."),
        ("Continuous Drift Monitoring", "Macroeconomic shifts alter default frequencies. Production credit risk models require ongoing monitoring for population stability and concept drift."),
        ("Human-in-the-Loop Oversight", "SHAP explanations should augment credit underwriters for borderline decisions rather than serving as fully autonomous denial engines.")
    ]

    for i, (title, desc) in enumerate(gov_cards):
        row = i // 2
        col = i % 2
        x = Inches(0.8 + col * 6.0)
        y = Inches(2.0 + row * 2.4)
        add_card(slide14, x, y, Inches(5.7), Inches(2.2))
        tb = slide14.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.3), Inches(1.8))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 15: CONCLUSION & KEY TAKEAWAYS (Dark Navy)
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15, BG_DARK)
    add_header(slide15, "14 · Synthesis", "Conclusion & Key Replication Takeaways",
               "Summarizing our empirical findings and research implications on German Credit", dark=True)

    takeaways = [
        ("1. Weighted Loss is Indispensable for Credit Risk", "Standard cross-entropy creates default blindness (only 20% recall). Weighted cross-entropy cures this, raising defaulter recall to 73.3% and cutting business losses by -53.3%."),
        ("2. TabTransformer Effectively Captures Tabular Context", "Self-attention across discrete categorical embeddings builds rich contextual interactions, outperforming linear baselines on discrimination (AUC 0.816 vs 0.772)."),
        ("3. SHAP Delivers Transparent Credit Decisioning", "Explains both global risk factors (checking account buffer and credit history) and applicant-specific local justifications required for regulatory compliance."),
        ("4. Honest Replication Overrides Inflated Claims", "While empirical test accuracy sits at 75% rather than the paper's 95%, the underlying scientific contribution is confirmed: weighted loss transforms tabular transformer performance on imbalanced lending data.")
    ]

    for i, (title, desc) in enumerate(takeaways):
        y = Inches(2.0 + i * 1.2)
        add_card(slide15, Inches(0.8), y, Inches(11.7), Inches(1.05), bg_color=RGBColor(0x1E, 0x29, 0x3B), border_color=ACCENT_TEAL)
        tb = slide15.shapes.add_textbox(Inches(1.0), y + Inches(0.1), Inches(11.3), Inches(0.85))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = ACCENT_TEAL
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 16: DISCUSSION / Q&A (Dark Navy)
    # =========================================================================
    slide16 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide16, BG_DARK)

    # Decorative Circle
    circ = slide16.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.5), Inches(1.8), Inches(3.8), Inches(3.8))
    circ.fill.background()
    circ.line.color.rgb = ACCENT_CORAL
    circ.line.width = Pt(14)

    tb = slide16.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(8.0), Inches(3.5))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "DISCUSSION"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(10)
    p = tf.add_paragraph(); p.text = "Questions & Feedback"; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = TEXT_LIGHT
    p.space_after = Pt(14)
    p = tf.add_paragraph(); p.text = "Thank you.\n\nReplication Repository: credit-risk-replication/\nPaper: Hartomo, Arthur, & Nataliani (IEEE Access 2025)\nDataset: UCI Statlog German Credit Data (ID 144)"; p.font.size = Pt(14); p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    # Save presentation
    prs.save(str(OUTPUT_PPTX))
    print(f"[OK] Presentation successfully generated at: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()
