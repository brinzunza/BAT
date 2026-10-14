"""Generates high-end PowerPoint pitch deck (.pptx) and Academic Research Paper (.docx).
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import docx
from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

OUTPUT_DIR = "/Users/brunoinzunza/Documents/github/PRODLIVETRADING"
PPTX_PATH = os.path.join(OUTPUT_DIR, "APEX_QUANTITATIVE_PITCH_DECK.pptx")
DOCX_PATH = os.path.join(OUTPUT_DIR, "DTFX_OMEGA_QUANTITATIVE_RESEARCH_PAPER.docx")

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE POWERPOINT PITCH DECK (.pptx)
# ─────────────────────────────────────────────────────────────────────────────
def create_pitch_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    BG_COLOR = RGBColor(15, 23, 42)       # Slate 900 #0F172A
    CARD_BG = RGBColor(30, 41, 59)        # Slate 800 #1E293B
    TEXT_WHITE = RGBColor(248, 250, 252)  # Slate 50 #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400 #94A3B8
    CYAN = RGBColor(6, 182, 212)          # Cyan 500 #06B6D4
    GOLD = RGBColor(245, 158, 11)         # Amber 500 #F59E0B
    GREEN = RGBColor(34, 197, 94)         # Green 500 #22C55E

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="APEX QUANTITATIVE CAPITAL"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_cat = cat_box.text_frame
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CYAN

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    # ── SLIDE 1: Title Slide ──
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)
    
    # Accent top border
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.1))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = CYAN
    top_bar.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "APEX QUANTITATIVE CAPITAL"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "DTFX Omega Systematic Strategy: Market-Structure Alpha"
    p2.font.size = Pt(22)
    p2.font.color.rgb = CYAN
    p2.space_after = Pt(24)

    p3 = tf.add_paragraph()
    p3.text = "• 22 Consecutive Profitable Years (2005 – 2026) with Zero Losing Calendar Years\n• 60.1% Win Rate Across 23,269 Live-Simulated Executions (+4,472 R Cumulative)\n• 93.5% Profitable Monthly Horizons & 100% Expanding Walk-Forward OOS Success\n• Non-Correlated Macro Alpha with Passive Zero-Spread Limit Execution"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_after = Pt(20)

    # ── SLIDE 2: Executive Summary ──
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(s2, "Executive Summary & Strategy Profile")

    # 4 Key Stat Cards
    stats = [
        ("22 / 22 Years", "100.0% Green Years", "Zero negative calendar years across 2005-2026 sample", GOLD),
        ("60.1% Win Rate", "1.51 Profit Factor", "+0.203 R Expectancy across 23,269 total trades", GREEN),
        ("6.53 Sharpe", "6.36 HAC-Adjusted", "Tightly bounded daily compounding risk-adjusted returns", CYAN),
        ("93.5% Months", "243 / 260 Profitable", "Extremely steady right-skewed monthly distributions", GOLD),
    ]

    for idx, (val, sub, desc, color) in enumerate(stats):
        col = idx % 4
        x = Inches(0.8 + col * 2.95)
        y = Inches(1.5)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(2.8), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = RGBColor(51, 65, 85)

        tb = s2.shapes.add_textbox(x + Inches(0.15), y + Inches(0.2), Inches(2.5), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = color
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = TEXT_WHITE
        p_sub.space_after = Pt(8)

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = TEXT_MUTED

    # Bottom summary card
    b_card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.0), Inches(11.7), Inches(2.8))
    b_card.fill.solid()
    b_card.fill.fore_color.rgb = CARD_BG
    b_card.line.color.rgb = RGBColor(51, 65, 85)
    
    b_tb = s2.shapes.add_textbox(Inches(1.1), Inches(4.2), Inches(11.1), Inches(2.4))
    b_tf = b_tb.text_frame
    b_tf.word_wrap = True
    
    p = b_tf.paragraphs[0]
    p.text = "Core Quantitative Thesis & Structural Edge:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.space_after = Pt(6)

    p_body = b_tf.add_paragraph()
    p_body.text = (
        "• Market-Structure Inefficiency: Traditional algorithms cross spreads prematurely upon supply/demand formation. DTFX Omega enforces a strict 3.37-hour maturation filter, eliminating exploratory retail wicks.\n"
        "• Passive Execution Advantage: Entries and primary exits are executed as resting limit orders at structural 50% retracements, capturing zero-spread executions.\n"
        "• Session Liquidity Concentration: Gated strictly to institutional volume rebalancing windows (SOMEGA), lifting standalone win rates to 66.9% - 69.6% across FX majors.\n"
        "• Dynamic Multi-Touch Compounding: Unbreached institutional zones are re-armed for up to 4 consecutive touches, extracting 56% more cumulative alpha."
    )
    p_body.font.size = Pt(11)
    p_body.font.color.rgb = TEXT_MUTED

    # ── SLIDE 3: Periodicity Performance ──
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(s3, "Multi-Decade Periodicity Performance (2005 – 2026)")

    # Table of periodicity
    rows, cols = 6, 6
    tbl_shape = s3.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(4.8))
    tbl = tbl_shape.table

    col_widths = [Inches(2.3), Inches(1.8), Inches(2.0), Inches(1.8), Inches(1.8), Inches(2.0)]
    for idx, w in enumerate(col_widths):
        tbl.columns[idx].width = w

    headers = ["Time Horizon", "Total Sample", "Profitable Units", "Win Rate %", "Profit Factor", "Average Return"]
    table_data = [
        ["Individual Trades", "23,269 trades", "13,993 wins", "60.1%", "1.51 PF", "+0.203 R / trade"],
        ["Trading Days", "5,228 days", "3,055 green days", "58.4%", "1.38 : 1", "+0.90 R / day"],
        ["Trading Weeks", "1,116 weeks", "863 green weeks", "77.3%", "2.13 : 1", "+4.23 R / week"],
        ["Trading Months", "260 months", "243 green months", "93.5%", "3.92 : 1", "+18.1 R / month"],
        ["Trading Years", "22 years", "22 green years", "100.0%", "inf (0 Losses)", "+214.4 R / year"]
    ]

    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(14, 116, 144)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = TEXT_WHITE

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else RGBColor(39, 52, 75)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.size = Pt(11)
                p.font.color.rgb = GOLD if c_idx == 4 and r_idx == 4 else (GREEN if c_idx in [3, 4] else TEXT_WHITE)

    # ── SLIDE 4: Walk-Forward Out-Of-Sample Validation ──
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(s4, "12-Year Expanding-Window Walk-Forward Out-Of-Sample (OOS) Scorecard")

    wf_data = [
        ["2015", "2009 - 2014", "1,135", "60.5%", "1.53", "+239.0 R", "6.94", "13.2%"],
        ["2016", "2009 - 2015", "1,110", "57.6%", "1.36", "+168.0 R", "5.10", "10.7%"],
        ["2017", "2009 - 2016", "1,221", "55.9%", "1.27", "+143.0 R", "3.86", "16.8%"],
        ["2018", "2009 - 2017", "1,386", "59.8%", "1.49", "+272.0 R", "6.54", "9.7%"],
        ["2019", "2009 - 2018", "1,253", "58.3%", "1.40", "+207.0 R", "5.36", "12.3%"],
        ["2020", "2009 - 2019", "1,595", "57.2%", "1.34", "+231.0 R", "5.22", "15.3%"],
        ["2021", "2009 - 2020", "1,426", "65.4%", "1.89", "+438.0 R", "10.37", "9.6%"],
        ["2022", "2009 - 2021", "1,409", "61.9%", "1.62", "+335.0 R", "8.37", "10.5%"],
        ["2023", "2009 - 2022", "1,291", "61.1%", "1.57", "+287.0 R", "6.87", "14.2%"],
        ["2024", "2009 - 2023", "1,274", "65.3%", "1.88", "+390.0 R", "10.26", "9.6%"],
        ["2025", "2009 - 2024", "1,469", "62.4%", "1.66", "+365.0 R", "7.93", "9.7%"],
        ["2026 YTD", "2009 - 2025", "897", "62.5%", "1.67", "+225.0 R", "7.86", "16.6%"],
    ]

    tbl_shape4 = s4.shapes.add_table(13, 8, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4))
    tbl4 = tbl_shape4.table
    wf_headers = ["OOS Year", "Anchor Window", "OOS Trades", "OOS Win %", "OOS PF", "OOS Return", "Daily Sharpe", "Max DD (1%)"]

    for c_idx, h in enumerate(wf_headers):
        cell = tbl4.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(14, 116, 144)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = TEXT_WHITE

    for r_idx, row in enumerate(wf_data):
        for c_idx, val in enumerate(row):
            cell = tbl4.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else RGBColor(39, 52, 75)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.size = Pt(9)
                p.font.color.rgb = GREEN if c_idx in [3, 4] else TEXT_WHITE

    # ── SLIDE 5: Microstructure Capacity & Adverse Selection ──
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(s5, "Market Microstructure Capacity & Adverse Selection Modeling")

    cap_data = [
        ["$1.0M", "$180,000", "98.8%", "99.4%", "0.08 pips", "1.48", "+0.193 R", "$454,307 / yr", "OPTIMAL (Zero Drag)"],
        ["$2.5M", "$450,000", "96.8%", "98.5%", "0.17 pips", "1.45", "+0.182 R", "$1,054,794 / yr", "OPTIMAL (Zero Drag)"],
        ["$5.0M", "$900,000", "93.2%", "97.0%", "0.31 pips", "1.39", "+0.163 R", "$1,839,741 / yr", "EXCELLENT (Fund Tier)"],
        ["$7.5M", "$1,350,000", "89.4%", "95.5%", "0.44 pips", "1.34", "+0.144 R", "$2,356,127 / yr", "EXCELLENT (Fund Tier)"],
        ["$10.0M", "$1,800,000", "85.6%", "94.0%", "0.57 pips", "1.28", "+0.123 R", "$2,606,558 / yr", "EXCELLENT (Fund Tier)"],
        ["$12.0M", "$2,160,000", "82.4%", "92.8%", "0.66 pips", "1.24", "+0.106 R", "$2,617,623 / yr", "🎯 PEAK EXTRACTION"],
        ["$14.0M", "$2,520,000", "79.3%", "91.6%", "0.75 pips", "1.19", "+0.088 R", "$2,462,477 / yr", "VIABLE (Prime Broker)"],
        ["$20.0M", "$3,600,000", "69.5%", "88.0%", "1.02 pips", "1.06", "+0.028 R", "$1,020,330 / yr", "MARGINAL CAPACITY"],
    ]

    tbl_shape5 = s5.shapes.add_table(9, 9, Inches(0.8), Inches(1.4), Inches(11.7), Inches(4.5))
    tbl5 = tbl_shape5.table
    cap_headers = ["Order Size", "Eq AUM (1%)", "P(fill|Win)", "P(fill|Loss)", "Avg Slip", "Profit Factor", "Avg R", "Annual PnL", "Feasibility Status"]

    for c_idx, h in enumerate(cap_headers):
        cell = tbl5.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(14, 116, 144)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(9)
            p.font.bold = True
            p.font.color.rgb = TEXT_WHITE

    for r_idx, row in enumerate(cap_data):
        for c_idx, val in enumerate(row):
            cell = tbl5.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else RGBColor(39, 52, 75)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.size = Pt(9)
                p.font.color.rgb = GOLD if "PEAK" in val or c_idx == 7 and r_idx == 5 else (GREEN if c_idx in [5, 7] else TEXT_WHITE)

    # Note box
    n_box = s5.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8))
    n_tf = n_box.text_frame
    p_n = n_tf.paragraphs[0]
    p_n.text = "Disclaimer: Capacity estimates are simulated microstructure scenario models conditional on historical L2 depth assumptions; they have not yet been validated through live institutional execution. The optimal execution sweet-spot is $1.0M - $12.0M notional ($180k - $2.16M AUM)."
    p_n.font.size = Pt(10)
    p_n.font.italic = True
    p_n.font.color.rgb = TEXT_MUTED

    # ── SLIDE 6: Investment Offerings & Architecture ──
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(s6, "Institutional Offerings & Production Deployment")

    # Left: Offerings
    c_left = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = CARD_BG
    c_left.line.color.rgb = RGBColor(51, 65, 85)

    tb_l = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "OPTION A: SEPARATELY MANAGED ACCOUNT (SMA)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.space_after = Pt(8)

    p_body = tf_l.add_paragraph()
    p_body.text = (
        "• Custody: 100% Investor Custody at OANDA / Interactive Brokers\n"
        "• Minimum Investment: $25,000 USD\n"
        "• Liquidity: Instant Daily Liquidity (Investor retains withdrawal control)\n"
        "• Management Fee: 0.0% (Zero Fixed Overhead)\n"
        "• Performance Fee: 25.0% with Strict High-Water Mark\n\n"
        "OPTION B: PRIVATE LP FUND / SYNDICATE\n"
        "• Structure: Delaware / Cayman Pooled Fund\n"
        "• Minimum Investment: $100,000 USD\n"
        "• Management Fee: 1.5% | Performance Fee: 20.0% HWM\n"
        "• Liquidity: Monthly Redemptions (15-day notice)"
    )
    p_body.font.size = Pt(10.5)
    p_body.font.color.rgb = TEXT_WHITE

    # Right: Tech Architecture
    c_right = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = CARD_BG
    c_right.line.color.rgb = RGBColor(51, 65, 85)

    tb_r = s6.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "24/7 PRODUCTION CLOUD ARCHITECTURE"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p.space_after = Pt(8)

    p_body2 = tf_r.add_paragraph()
    p_body2.text = (
        "• Cloud Infrastructure: Dedicated Linux VM with 24/7 systemd daemon\n"
        "• Order Execution: OANDA v20 REST / WebSocket API (<30ms execution)\n"
        "• Dual-Account FIFO Bypass: Long (002) & Short (003) segregated sub-accounts\n"
        "• Database Persistence: SQLite WAL Mode with automatic crash recovery\n"
        "• Real-Time Monitoring: Bi-directional Telegram Bot (@brinztradingbot)\n"
        "• Parameter Freeze: Formally frozen under APEX v1.0 (PROD)\n\n"
        "CONTACT & DUE DILIGENCE:\n"
        "• Research Repo: github.com/brinzunza/data-research\n"
        "• Production Repo: github.com/brinzunza/PRODLIVETRADING\n"
        "• Telemetry Bot: @brinztradingbot"
    )
    p_body2.font.size = Pt(10.5)
    p_body2.font.color.rgb = TEXT_WHITE

    prs.save(PPTX_PATH)
    print(f"✅ PowerPoint pitch deck created successfully: {PPTX_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE QUANTITATIVE RESEARCH PAPER (.docx)
# ─────────────────────────────────────────────────────────────────────────────
def create_research_paper_docx():
    doc = docx.Document()

    # Page Margins: 1 inch all around
    sections = doc.sections
    for s in sections:
        s.top_margin = DocxInches(1.0)
        s.bottom_margin = DocxInches(1.0)
        s.left_margin = DocxInches(1.0)
        s.right_margin = DocxInches(1.0)

    # Style: Calibri / Times New Roman
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = DocxPt(11)
    font.color.rgb = DocxRGBColor(30, 41, 59)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Extracting Systematic Alpha from Microstructure Liquidity Imbalances: An Empirical Study of the DTFX Omega Market-Structure Framework (2005–2026)")
    run_title.font.size = DocxPt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = DocxRGBColor(15, 23, 42)

    # Author
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_auth = p_auth.add_run("Bruno Inzunza\nApex Quantitative Research & Systems Engineering\nAugust 2026")
    run_auth.font.size = DocxPt(11)
    run_auth.font.italic = True
    run_auth.font.color.rgb = DocxRGBColor(100, 116, 139)

    doc.add_paragraph().paragraph_format.space_after = DocxPt(12)

    # Abstract Box
    tbl_abs = doc.add_table(rows=1, cols=1)
    tbl_abs.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_abs = tbl_abs.cell(0, 0)
    cell_abs.width = DocxInches(6.5)
    
    # Border styling
    tcPr = cell_abs._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F1F5F9"/>')
    tcPr.append(shd)

    p_ab = cell_abs.paragraphs[0]
    p_ab.paragraph_format.left_indent = DocxInches(0.15)
    p_ab.paragraph_format.right_indent = DocxInches(0.15)
    p_ab.paragraph_format.space_before = DocxPt(8)
    p_ab.paragraph_format.space_after = DocxPt(8)

    r_ab_head = p_ab.add_run("Abstract—")
    r_ab_head.font.bold = True
    r_ab_head.font.size = DocxPt(10)
    
    r_ab_body = p_ab.add_run(
        "We investigate the persistence and structural mechanics of intra-day liquidity imbalances in high-volume foreign exchange markets over a continuous 22-year historical horizon (2005–2026). Traditional mean-reversion and momentum models suffer severe transaction drag and parameter fragility due to premature retracement traps and bid-ask spread friction. We introduce DTFX Omega, an autonomous quantitative market-structure strategy that integrates: (i) strict causal zigzag pivot filtration, (ii) a causal 3.37-hour maturation trap barrier, (iii) institutional session liquidity gating (SOMEGA), (iv) zero-spread hybrid limit order exits, and (v) multi-touch zone compounding. Evaluating over 6.3 million 1-minute price bars across 23,269 executed trades, the strategy produces an empirical win rate of 60.1%, a Profit Factor of 1.51, an annualized Daily Sharpe of 6.53 (Newey-West HAC-adjusted: 6.36), and 22 consecutive profitable years with zero negative annual returns. We subject the framework to rigorous out-of-sample stress testing, including a 12-year expanding-window walk-forward validation (100% profitable OOS years), parameter sensitivity sweeps (±30% invariant plateau), 10,000-sample bootstrap confidence intervals (5th percentile PF = 1.47), synthetic randomness permutations (Z = +30.53σ, p < 10^-4), and an empirical market microstructure capacity model incorporating asymmetric limit order adverse selection."
    )
    r_ab_body.font.size = DocxPt(10)

    doc.add_paragraph().paragraph_format.space_after = DocxPt(14)

    def add_sec_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = DocxPt(14)
        p.paragraph_format.space_after = DocxPt(4)
        run = p.add_run(text)
        run.font.size = DocxPt(13)
        run.font.bold = True
        run.font.color.rgb = DocxRGBColor(14, 116, 144)

    # 1. Introduction
    add_sec_heading("1. Introduction & Theoretical Motivation")
    doc.add_paragraph(
        "Modern algorithmic execution across centralized and decentralized FX limit order books (LOB) is characterized by intense high-frequency competition. Standard algorithmic strategies tend to decay rapidly due to two primary failure modes:\n\n"
        "1. The Retracement Liquidity Trap: Newly established supply and demand levels (order blocks) attract significant retail and algorithmic participation immediately upon creation. During the first 1 to 2 hours of zone formation, price action frequently exhibits noisy exploratory wicks ('stop hunts'), resulting in premature execution and stop-out before directional continuation occurs.\n\n"
        "2. Aggressive Order Execution Drag: Crossing the bid-ask spread twice via market orders creates an insurmountable drag of 0.8–2.0 pips per round-trip trade, which rapidly exhausts the edge of intraday trading systems.\n\n"
        "To resolve these systemic inefficiencies, we formulate the DTFX Omega framework, demonstrating that market-structure boundaries exhibit strong, exploitable mean-reversion behavior conditional on specific temporal maturation constraints and institutional liquidity windows."
    )

    # 2. Methodology
    add_sec_heading("2. Mathematical Methodology & Strategy Architecture")
    doc.add_paragraph(
        "Let {p_t} denote the time series of 1-minute FX prices. Major structural swing pivots are identified via strict causal zigzag filtration requiring a minimum retracement of 2.0x ATR. Symmetrical risk boundaries are derived from the zone midpoint:\n\n"
        "• Zone Width: W_z = P_high - P_low\n"
        "• Zone Midpoint: Z_mid = (P_high + P_low) / 2\n"
        "• Risk Distance: R_z = N_mult * (W_z / 2), where N_mult = 3.343496\n"
        "• Volatility Floor: R_z / Z_mid >= 0.000170\n\n"
        "Temporal Maturation Constraint:\n"
        "Zones touched prematurely (Delta tau < 202.0 minutes / 3.367 hours) are disqualified outright. Disqualifying premature touches eliminates exploratory retail traps while preserving genuine institutional re-accumulation orders."
    )

    # 3. Empirical Track Record Table
    add_sec_heading("3. Multi-Decade Periodicity Performance (2005–2026)")
    doc.add_paragraph("Table 1 summarizes the empirical performance metrics across 23,269 live-simulated trades over 22 continuous historical years:")

    t1 = doc.add_table(rows=6, cols=6)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_headers = ["Time Horizon", "Total Sample", "Profitable Periods", "Win Rate (%)", "Profit Factor", "Average Net Return"]
    t1_rows = [
        ["Individual Trades", "23,269 trades", "13,993 wins", "60.1%", "1.51 PF", "+0.203 R / trade"],
        ["Trading Days", "5,228 days", "3,055 green days", "58.4%", "1.38 : 1", "+0.90 R / day"],
        ["Trading Weeks", "1,116 weeks", "863 green weeks", "77.3%", "2.13 : 1", "+4.23 R / week"],
        ["Trading Months", "260 months", "243 green months", "93.5%", "3.92 : 1", "+18.1 R / month"],
        ["Trading Years", "22 years", "22 green years", "100.0%", "inf (0 Losses)", "+214.4 R / year"]
    ]

    for c_idx, h in enumerate(t1_headers):
        cell = t1.cell(0, c_idx)
        cell.text = h
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="0E7490"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        for p in cell.paragraphs:
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = DocxRGBColor(255, 255, 255)
            p.runs[0].font.size = DocxPt(9.5)

    for r_idx, row in enumerate(t1_rows):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx + 1, c_idx)
            cell.text = val
            fill_c = "F8FAFC" if r_idx % 2 == 0 else "F1F5F9"
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_c}"/>')
            cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.runs[0].font.size = DocxPt(9.5)

    doc.add_paragraph().paragraph_format.space_after = DocxPt(12)

    # 4. Walk-Forward OOS Table
    add_sec_heading("4. Expanding-Window Walk-Forward Out-Of-Sample Validation")
    doc.add_paragraph("Table 2 presents the 12-year expanding-window walk-forward out-of-sample scorecard under fixed frozen parameters:")

    t2 = doc.add_table(rows=13, cols=8)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["OOS Year", "Train Window", "OOS Trades", "OOS Win %", "OOS PF", "OOS Return", "Daily Sharpe", "Max DD (1%)"]
    t2_data = [
        ["2015", "2009 - 2014", "1,135", "60.5%", "1.53", "+239.0 R", "6.94", "13.2%"],
        ["2016", "2009 - 2015", "1,110", "57.6%", "1.36", "+168.0 R", "5.10", "10.7%"],
        ["2017", "2009 - 2016", "1,221", "55.9%", "1.27", "+143.0 R", "3.86", "16.8%"],
        ["2018", "2009 - 2017", "1,386", "59.8%", "1.49", "+272.0 R", "6.54", "9.7%"],
        ["2019", "2009 - 2018", "1,253", "58.3%", "1.40", "+207.0 R", "5.36", "12.3%"],
        ["2020", "2009 - 2019", "1,595", "57.2%", "1.34", "+231.0 R", "5.22", "15.3%"],
        ["2021", "2009 - 2020", "1,426", "65.4%", "1.89", "+438.0 R", "10.37", "9.6%"],
        ["2022", "2009 - 2021", "1,409", "61.9%", "1.62", "+335.0 R", "8.37", "10.5%"],
        ["2023", "2009 - 2022", "1,291", "61.1%", "1.57", "+287.0 R", "6.87", "14.2%"],
        ["2024", "2009 - 2023", "1,274", "65.3%", "1.88", "+390.0 R", "10.26", "9.6%"],
        ["2025", "2009 - 2024", "1,469", "62.4%", "1.66", "+365.0 R", "7.93", "9.7%"],
        ["2026 YTD", "2009 - 2025", "897", "62.5%", "1.67", "+225.0 R", "7.86", "16.6%"],
    ]

    for c_idx, h in enumerate(t2_headers):
        cell = t2.cell(0, c_idx)
        cell.text = h
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="0E7490"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        for p in cell.paragraphs:
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = DocxRGBColor(255, 255, 255)
            p.runs[0].font.size = DocxPt(9)

    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx + 1, c_idx)
            cell.text = val
            fill_c = "F8FAFC" if r_idx % 2 == 0 else "F1F5F9"
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_c}"/>')
            cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.runs[0].font.size = DocxPt(9)

    doc.add_paragraph().paragraph_format.space_after = DocxPt(12)

    # 5. Statistical Rigor
    add_sec_heading("5. Statistical Rigor & Hypothesis Testing")
    doc.add_paragraph(
        "• 10,000-Iteration Bootstrap: Mean R/trade 95% CI is [+0.187 R, +0.213 R]. 5th percentile conservative Profit Factor lower bound is 1.47.\n"
        "• Multi-Scale Sharpe: Trade Sharpe = 7.43, Daily Compounding Sharpe = 6.53, Newey-West HAC Adjusted Daily Sharpe = 6.36, Monthly Sharpe = 5.94.\n"
        "• Randomness Null Hypothesis: Zero exceedances observed across 10,000 random coin-flip portfolios (Empirical p < 10^-4, Theoretical Z = +30.53 standard deviations above randomness)."
    )

    # 6. Microstructure Capacity
    add_sec_heading("6. Market Microstructure Capacity & Adverse Selection")
    doc.add_paragraph(
        "In FX trading, Notional Exposure != Risk Capital. With an average 18-pip stop (0.0018), 1% equity risk corresponds to 5.55x notional leverage.\n\n"
        "Asymmetric Adverse Selection Modeling on Passive Limit Orders:\n"
        "Resting limit orders experience asymmetric fill probabilities conditional on post-touch price velocity. On winning trades (price grazes 50% limit and reverses), queue priority drops fill rate. On losing trades (price slices through zone to stop), volume guarantees near-100% fills.\n\n"
        "Factoring in adverse selection, the empirical dollar profit extraction curve peaks at $12.0M notional order size ($2.16M AUM at 1% risk), extracting $2.62M/year net profit. Up to $2.5M AUM ($14M notional), the strategy operates with minimal execution degradation.\n\n"
        "Disclaimer: Capacity estimates are simulated microstructure scenario models conditional on historical L2 depth assumptions; they have not yet been validated through live institutional execution."
    )

    # 7. Production Architecture
    add_sec_heading("7. Production Deployment & Strategy Freeze Manifest")
    doc.add_paragraph(
        "The DTFX Omega framework is deployed 24/7 on a dedicated Linux Cloud VM via an event-driven Python engine with OANDA v20 WebSocket streaming (<30ms response), SQLite WAL database persistence, and dual-account FIFO bypass.\n\n"
        "To prevent data-snooping and hindsight overfitting, all parameters are formally frozen under APEX v1.0 (PROD):\n"
        "• MIN_RETRACEMENT = 2.0\n"
        "• N_MULTIPLIER = 3.343496\n"
        "• MIN_FILL_AGE = 3.366682 hours (202 minutes)\n"
        "• MIN_RISK_PCT = 0.000170\n"
        "• MAX_ZONE_TOUCHES = 4\n"
        "• HYBRID_BACKSTOP = 1.5x\n"
        "• SOMEGA SESSIONS = EURGBP (18-23), GBPUSD (20-22), USDJPY (19-22)"
    )

    doc.save(DOCX_PATH)
    print(f"✅ Quantitative Research Paper (.docx) created successfully: {DOCX_PATH}")

if __name__ == "__main__":
    create_pitch_deck()
    create_research_paper_docx()
