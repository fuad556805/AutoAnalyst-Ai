import io
import base64
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re


def _clean_html(text):
    return re.sub(r'<[^>]+>', '', str(text))


def generate_pdf_report(meta: dict, dataset_name: str, ml_results: list,
                        charts_data: dict, feature_importance: list,
                        insights: list) -> bytes:
    W, H = A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title='AutoAnalyst AI — ML Report',
        author='AutoAnalyst AI',
    )

    BLACK  = colors.HexColor('#09090b')
    DARK   = colors.HexColor('#3f3f46')
    MUTED  = colors.HexColor('#71717a')
    MUTED2 = colors.HexColor('#a1a1aa')
    GREEN  = colors.HexColor('#16a34a')
    BORDER = colors.HexColor('#e4e4e7')
    BG     = colors.HexColor('#fafafa')
    WHITE  = colors.white

    sH2  = ParagraphStyle('H2',  fontSize=14, leading=18, textColor=BLACK, fontName='Helvetica-Bold', spaceBefore=18, spaceAfter=6)
    sH3  = ParagraphStyle('H3',  fontSize=11, leading=14, textColor=DARK,  fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4)
    sBod = ParagraphStyle('Bod', fontSize=9,  leading=14, textColor=DARK,  fontName='Helvetica', spaceAfter=4)
    sMut = ParagraphStyle('Mut', fontSize=8,  leading=11, textColor=MUTED, fontName='Helvetica')

    def table_style(header_color=BLACK):
        return TableStyle([
            ('BACKGROUND',   (0,0), (-1,0),  header_color),
            ('TEXTCOLOR',    (0,0), (-1,0),  WHITE),
            ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,-1), 8.5),
            ('LEADING',      (0,0), (-1,-1), 12),
            ('GRID',         (0,0), (-1,-1), 0.4, BORDER),
            ('LEFTPADDING',  (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
            ('TOPPADDING',   (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [BG, WHITE]),
        ])

    def hr():
        return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=4)

    def sp(n=10):
        return Spacer(1, n)

    def add_chart(story, b64_str, caption='', width_cm=13):
        try:
            img_bytes = io.BytesIO(base64.b64decode(b64_str))
            img = RLImage(img_bytes, width=width_cm*cm, height=width_cm*0.62*cm)
            story.append(img)
            if caption:
                story.append(Paragraph(caption, sMut))
            story.append(sp(6))
        except Exception:
            pass

    story = []
    pre = meta.get('preprocessing', {})
    now = datetime.datetime.now().strftime('%B %d, %Y')

    # ── TITLE PAGE ───────────────────────────────────────────────
    story.append(sp(40))
    story.append(Paragraph('AutoAnalyst AI', ParagraphStyle(
        'Cover', fontSize=28, leading=32, textColor=BLACK,
        fontName='Helvetica-Bold', alignment=TA_CENTER)))
    story.append(sp(6))
    story.append(Paragraph('Machine Learning Pipeline Report', ParagraphStyle(
        'CoverSub', fontSize=14, leading=18, textColor=MUTED,
        fontName='Helvetica', alignment=TA_CENTER)))
    story.append(sp(20))
    story.append(HRFlowable(width='60%', thickness=1.5, color=BLACK, hAlign='CENTER'))
    story.append(sp(20))

    meta_cover = [
        ['Dataset',      dataset_name],
        ['Target',       meta.get('target', 'N/A')],
        ['Problem Type', meta.get('problem_type', 'N/A').title()],
        ['Best Model',   meta.get('best_model_name', 'N/A')],
        [meta.get('metric_label', 'Score'), f"{meta.get('best_score', 'N/A')}%"],
        ['Report Date',  now],
    ]
    tm = Table(meta_cover, colWidths=[5*cm, 10*cm], hAlign='CENTER')
    tm.setStyle(TableStyle([
        ('FONTNAME',     (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',     (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,0), (-1,-1), 10),
        ('LEADING',      (0,0), (-1,-1), 15),
        ('TEXTCOLOR',    (0,0), (0,-1), MUTED),
        ('TEXTCOLOR',    (1,0), (1,-1), BLACK),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('LINEBELOW',    (0,0), (-1,-2), 0.3, BORDER),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
    ]))
    story.append(tm)
    story.append(PageBreak())

    # ── SECTION 1: DATASET OVERVIEW ──────────────────────────────
    story.append(Paragraph('1. Dataset Overview', sH2))
    story.append(hr())
    ov_data = [
        ['Property', 'Value', 'Notes'],
        ['Original Rows',       str(pre.get('original_rows', 'N/A')), 'Total records in uploaded file'],
        ['Original Columns',    str(pre.get('original_cols', 'N/A')), 'Total features including target'],
        ['Duplicate Rows',      str(pre.get('duplicates_count', 'N/A')), 'Identical rows detected'],
        ['Rows After Cleaning', str(pre.get('rows_after_clean', 'N/A')), 'After null-target removal'],
        ['Rows Dropped',        str(pre.get('rows_dropped', 'N/A')), 'Rows with missing target'],
        ['Columns Used',        str(pre.get('cols_after', 'N/A')), 'After dropping uninformative columns'],
        ['Training Samples',    str(pre.get('n_train', 'N/A')), '80% split (seed=42)'],
        ['Test Samples',        str(pre.get('n_test',  'N/A')), '20% held-out evaluation'],
    ]
    story.append(Table(ov_data, colWidths=[4.5*cm, 3.5*cm, 8*cm], style=table_style()))

    dropped = pre.get('cols_dropped', [])
    if dropped:
        story.append(sp())
        story.append(Paragraph('Columns Removed', sH3))
        dc_data = [['#', 'Column', 'Reason']]
        for i, col in enumerate(dropped, 1):
            dc_data.append([str(i), col, 'High missing / constant / identifier'])
        story.append(Table(dc_data, colWidths=[1*cm, 7*cm, 8*cm], style=table_style()))

    story.append(PageBreak())

    # ── SECTION 2: MISSING VALUES ────────────────────────────────
    story.append(Paragraph('2. Data Quality & Missing Values', sH2))
    story.append(hr())
    null_info = pre.get('null_info', {})
    if null_info:
        story.append(Paragraph(
            f'Missing values found in <strong>{len(null_info)}</strong> column(s). '
            'Numeric → Median imputation. Categorical → Mode imputation.', sBod))
        story.append(sp())
        mv_data = [['Column', 'Type', 'Missing #', 'Missing %', 'Strategy']]
        for col, info in null_info.items():
            dtype_str = info.get('dtype', 'N/A')
            strategy = 'Median' if ('int' in dtype_str or 'float' in dtype_str) else 'Mode'
            mv_data.append([col, dtype_str, str(info.get('count', 0)),
                             f"{info.get('pct', 0)}%", strategy])
        story.append(Table(mv_data, colWidths=[4*cm, 3*cm, 2.5*cm, 2.5*cm, 4*cm], style=table_style()))
    else:
        story.append(Paragraph('✓ No missing values detected.', ParagraphStyle(
            'Ok', fontSize=10, leading=14, textColor=GREEN, fontName='Helvetica-Bold')))
    story.append(PageBreak())

    # ── SECTION 3: FEATURE ENGINEERING ──────────────────────────
    story.append(Paragraph('3. Feature Engineering & Encoding', sH2))
    story.append(hr())
    lm = meta.get('label_mappings', {})
    feature_names = meta.get('feature_names', [])
    story.append(Paragraph(
        f'<strong>{len(feature_names)}</strong> feature(s) used. '
        'StandardScaler applied to all. Categorical features label-encoded first.', sBod))
    story.append(sp())
    feat_data = [['#', 'Feature', 'Type', 'Encoding']]
    for i, feat in enumerate(feature_names, 1):
        feat_type = 'Categorical' if feat in lm else 'Numeric'
        enc = 'Label Encoding + StandardScaler' if feat in lm else 'StandardScaler only'
        feat_data.append([str(i), feat, feat_type, enc])
    story.append(Table(feat_data, colWidths=[1*cm, 6*cm, 3*cm, 6*cm], style=table_style()))
    story.append(PageBreak())

    # ── SECTION 4: MODEL TRAINING ────────────────────────────────
    story.append(Paragraph('4. Model Training & Selection', sH2))
    story.append(hr())
    problem_type = meta.get('problem_type', 'classification')
    story.append(Paragraph(
        f'Problem: <strong>{problem_type.title()}</strong>. '
        f'Selection metric: <strong>{meta.get("metric_label","Score")}</strong> on 20% test set.', sBod))
    story.append(sp())
    if ml_results:
        r_data = [['Rank', 'Model', 'Score (%)', 'Status']]
        for i, r in enumerate(sorted(ml_results, key=lambda x: x['score'], reverse=True), 1):
            is_best = r['name'] == meta.get('best_model_name')
            r_data.append([str(i), r['name'], f"{r['score']}%",
                           '★ Best' if is_best else '—'])
        story.append(Table(r_data, colWidths=[1.5*cm, 8*cm, 4*cm, 3*cm], style=table_style()))

    extra = meta.get('extra_metrics', {})
    if extra:
        story.append(sp())
        story.append(Paragraph('Additional Metrics (Best Model)', sH3))
        em_data = [['Metric', 'Value']]
        for k, v in extra.items():
            unit = '%' if isinstance(v, float) and v > 1 else ''
            em_data.append([k, f'{v}{unit}'])
        story.append(Table(em_data, colWidths=[5*cm, 5*cm], style=table_style()))
    story.append(PageBreak())

    # ── SECTION 5: VISUAL ANALYSIS ───────────────────────────────
    if charts_data:
        story.append(Paragraph('5. Visual Analysis', sH2))
        story.append(hr())
        if 'model_comparison' in charts_data:
            story.append(Paragraph('5.1 Model Comparison', sH3))
            add_chart(story, charts_data['model_comparison'],
                      f'Model comparison by {meta.get("metric_label","Score")}')
        if 'feature_importance' in charts_data:
            story.append(Paragraph('5.2 Feature Importance', sH3))
            add_chart(story, charts_data['feature_importance'], 'Feature importance scores')
        if 'confusion' in charts_data:
            story.append(Paragraph('5.3 Confusion Matrix', sH3))
            add_chart(story, charts_data['confusion'], 'Confusion matrix on test set', width_cm=9)
        if 'actual_vs_pred' in charts_data:
            story.append(Paragraph('5.3 Actual vs Predicted', sH3))
            add_chart(story, charts_data['actual_vs_pred'], 'Actual vs Predicted on test set')
        story.append(PageBreak())

    # ── SECTION 6: FEATURE IMPORTANCE TABLE ──────────────────────
    sec = 6
    if feature_importance:
        story.append(Paragraph(f'{sec}. Feature Importance Rankings', sH2))
        story.append(hr())
        fi_data = [['Rank', 'Feature', 'Score', 'Relative Strength']]
        max_imp = feature_importance[0]['importance'] if feature_importance else 1
        for i, item in enumerate(feature_importance, 1):
            pct = round(item['importance'] / max_imp * 100, 1)
            bars = '█' * int(pct / 10)
            fi_data.append([str(i), item['feature'], f"{item['importance']:.4f}", f"{bars} {pct}%"])
        story.append(Table(fi_data, colWidths=[1.5*cm, 7*cm, 4*cm, 4.5*cm], style=table_style()))
        story.append(PageBreak())
        sec += 1

    # ── SECTION 7: INSIGHTS ──────────────────────────────────────
    story.append(Paragraph(f'{sec}. Auto-Generated Insights', sH2))
    story.append(hr())
    if insights:
        for ins in insights:
            story.append(Paragraph(f'• {_clean_html(ins)}', sBod))
            story.append(sp(3))
    else:
        story.append(Paragraph('No insights generated.', sMut))

    story.append(sp())
    story.append(Paragraph('Recommendations', sH3))
    recs = [
        'Collect more data if model accuracy is below expectations.',
        'Apply Recursive Feature Elimination to reduce noise.',
        'Consider hyperparameter tuning (GridSearchCV) on the best model.',
        'Use k-fold cross-validation for a more robust performance estimate.',
        'If class imbalance exists, consider SMOTE or class_weight adjustments.',
    ]
    for rec in recs:
        story.append(Paragraph(f'→ {rec}', sBod))
        story.append(sp(2))

    doc.build(story)
    return buf.getvalue()
