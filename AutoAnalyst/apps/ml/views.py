import os
import io
import json
import base64
import datetime
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                              classification_report, confusion_matrix,
                              mean_absolute_error)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib


# ── Matplotlib helpers ─────────────────────────────────────────────────────────

CHART_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#e4e4e7',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#09090b',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#e4e4e7',
}

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


# ── Data helpers ───────────────────────────────────────────────────────────────

def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def _detect_problem_type(series):
    if not _is_numeric(series):
        return 'classification'
    n_unique = series.nunique()
    if n_unique <= 20:
        return 'classification'
    return 'regression'


def _clean_df(df, target_col):
    rows_before = len(df)
    df = df.dropna(subset=[target_col])

    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        missing_pct = df[col].isna().mean()
        if missing_pct > 0.80:
            cols_to_drop.append(col)
            continue
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
            continue
        if col.lower() in ('id', 'index', 'row', 'no', 'num', '#', 'uuid',
                            'email', 'phone', 'name', 'rowid'):
            cols_to_drop.append(col)
            continue
        if not _is_numeric(df[col]) and df[col].nunique() / max(len(df), 1) > 0.90:
            cols_to_drop.append(col)
            continue

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    rows_after = len(df)
    return df.copy(), rows_before, rows_after, cols_to_drop


def _preprocess(df, target_col):
    encoders = {}
    label_mappings = {}
    imputation_log = []   # track what was imputed

    # Step 1 — Impute BEFORE encoding
    for col in df.columns:
        if df[col].isna().any():
            n_missing = int(df[col].isna().sum())
            if _is_numeric(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                imputation_log.append({
                    'column': col,
                    'missing': n_missing,
                    'strategy': 'Median imputation',
                    'fill_value': round(float(fill_val), 4),
                })
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill)
                imputation_log.append({
                    'column': col,
                    'missing': n_missing,
                    'strategy': 'Mode imputation',
                    'fill_value': str(fill),
                })

    # Step 2 — Encode all non-numeric columns
    for col in df.columns:
        if not _is_numeric(df[col]):
            le = LabelEncoder()
            original_classes = sorted(df[col].astype(str).unique().tolist())
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            label_mappings[col] = list(le.classes_)

    # Step 3 — Final safety coerce
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, scaler, encoders, label_mappings, feature_names, imputation_log


# ── Chart generators ───────────────────────────────────────────────────────────

def _chart_confusion(y_test, preds, labels=None):
    cm = confusion_matrix(y_test, preds, labels=labels)
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        im = ax.imshow(cm, cmap='Greys')
        ax.set_xticks(range(len(cm)))
        ax.set_yticks(range(len(cm)))
        if labels is not None:
            ax.set_xticklabels([str(l) for l in labels], rotation=45, ha='right')
            ax.set_yticklabels([str(l) for l in labels])
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() * 0.5 else '#09090b', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_actual_vs_pred(y_test, preds):
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, preds, alpha=0.5, color='#18181b', s=20, zorder=3)
        mn = min(float(y_test.min()), float(preds.min()))
        mx = max(float(y_test.max()), float(preds.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        ax.legend(fontsize=8)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_feature_importance(feature_names, importances, top_n=10):
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.4)))
        bars = ax.barh(names, vals, color='#18181b', edgecolor='none', height=0.6)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        ax.set_xlim(0, max(vals) * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_model_comparison(names, scores, metric_label, best_name):
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.55)))
        colors = ['#16a34a' if n == best_name else '#18181b' for n in names]
        bars = ax.barh(names, scores, color=colors, edgecolor='none', height=0.55)
        ax.set_xlabel(f'{metric_label} (%)')
        ax.set_title('Model Comparison')
        ax.set_xlim(0, max(scores) * 1.15 if scores else 100)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val}%', va='center', fontsize=8, fontweight='bold')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return _fig_to_b64(fig)


# ── Views ──────────────────────────────────────────────────────────────────────

def train(request):
    path   = request.session.get('dataset_path')
    target = request.session.get('target_column')

    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset found. Please upload a file first.')
        return redirect('upload')
    if not target:
        messages.error(request, 'Please select a target column first.')
        return redirect('select_target')

    ext = path.rsplit('.', 1)[1].lower()
    try:
        df = pd.read_csv(path) if ext == 'csv' else pd.read_excel(path)
    except Exception as e:
        messages.error(request, f'Could not read dataset: {e}')
        return redirect('upload')

    if target not in df.columns:
        messages.error(request, f'Target column "{target}" not found.')
        return redirect('select_target')
    if len(df) < 5:
        messages.error(request, 'Dataset is too small (need at least 5 rows).')
        return redirect('upload')

    try:
        original_rows = len(df)
        original_cols = len(df.columns)
        duplicates_count = int(df.duplicated().sum())

        # Collect null info BEFORE cleaning
        null_info = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                pct = round(n / len(df) * 100, 2)
                null_info[col] = {'count': n, 'pct': pct,
                                  'dtype': str(df[col].dtype)}

        df, rows_before, rows_after, dropped_cols = _clean_df(df, target)
        problem_type = _detect_problem_type(df[target])

        X_train, X_test, y_train, y_test, scaler, encoders, label_mappings, feature_names, imputation_log = \
            _preprocess(df.copy(), target)
    except Exception as e:
        messages.error(request, f'Preprocessing error: {e}')
        return redirect('select_target')

    # ── Train models ──────────────────────────────────────────────────────────
    results = []
    charts_data = {}

    if problem_type == 'classification':
        metric_label = 'Accuracy'
        model_list = [
            ('Logistic Regression',    LogisticRegression(max_iter=2000, random_state=42)),
            ('Random Forest',          RandomForestClassifier(n_estimators=100, random_state=42)),
            ('K-Nearest Neighbors',    KNeighborsClassifier(n_neighbors=min(5, len(X_train)))),
            ('Support Vector Machine', SVC(random_state=42)),
        ]
        for name, model in model_list:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = round(accuracy_score(y_test, preds) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model,
                                 'preds': preds, 'y_test': y_test})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None})

        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]

        if best.get('preds') is not None:
            try:
                labels = sorted(y_test.unique())
                charts_data['confusion'] = _chart_confusion(
                    best['y_test'], best['preds'], labels=labels)
            except Exception:
                pass

        extra_metrics = {}
        if best.get('preds') is not None:
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                avg = 'binary' if len(y_test.unique()) == 2 else 'weighted'
                extra_metrics = {
                    'Precision': round(precision_score(y_test, best['preds'], average=avg, zero_division=0) * 100, 2),
                    'Recall':    round(recall_score(y_test,    best['preds'], average=avg, zero_division=0) * 100, 2),
                    'F1 Score':  round(f1_score(y_test,        best['preds'], average=avg, zero_division=0) * 100, 2),
                }
            except Exception:
                extra_metrics = {}

    else:
        metric_label = 'R² Score'
        model_list = [
            ('Linear Regression',       LinearRegression()),
            ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Ridge Regression',        Ridge()),
        ]
        for name, model in model_list:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                raw_r2 = r2_score(y_test, preds)
                score  = round(max(raw_r2, 0) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model,
                                 'preds': preds, 'y_test': y_test})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None})

        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]

        if best.get('preds') is not None:
            try:
                charts_data['actual_vs_pred'] = _chart_actual_vs_pred(
                    best['y_test'], best['preds'])
            except Exception:
                pass

        extra_metrics = {}
        if best.get('preds') is not None:
            try:
                extra_metrics = {
                    'MAE':  round(float(mean_absolute_error(y_test, best['preds'])), 4),
                    'RMSE': round(float(np.sqrt(mean_squared_error(y_test, best['preds']))), 4),
                    'MSE':  round(float(mean_squared_error(y_test, best['preds'])), 4),
                }
            except Exception:
                extra_metrics = {}

    # ── Feature importance ────────────────────────────────────────────────────
    feature_importance = []
    best_model = best['model']
    fi_chart = None

    if best_model is not None:
        try:
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                pairs = sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True)[:10]
                feature_importance = [
                    {'feature': f, 'importance': round(float(v), 4)} for f, v in pairs
                ]
                fi_chart = _chart_feature_importance(
                    [p[0] for p in pairs], [p[1] for p in pairs])
            elif hasattr(best_model, 'coef_'):
                coef = np.array(best_model.coef_).flatten()
                pairs = sorted(zip(feature_names, np.abs(coef)),
                               key=lambda x: x[1], reverse=True)[:10]
                feature_importance = [
                    {'feature': f, 'importance': round(float(v), 4)} for f, v in pairs
                ]
                fi_chart = _chart_feature_importance(
                    [p[0] for p in pairs], [p[1] for p in pairs])
        except Exception:
            pass

    if fi_chart:
        charts_data['feature_importance'] = fi_chart

    try:
        names  = [r['name'] for r in results]
        scores = [r['score'] for r in results]
        charts_data['model_comparison'] = _chart_model_comparison(
            names, scores, metric_label, best['name'])
    except Exception:
        pass

    # ── Auto insights ─────────────────────────────────────────────────────────
    insights = []
    try:
        insights.append(f"Best model: <strong>{best['name']}</strong> with {metric_label} = <strong>{best['score']}%</strong>")
        if len(results) > 1:
            worst = min(results, key=lambda x: x['score'])
            diff  = round(best['score'] - worst['score'], 2)
            if diff > 0:
                insights.append(f"Performance spread across models: <strong>{diff}%</strong> difference between best and worst model.")
        if dropped_cols:
            insights.append(f"Auto-cleaned: removed <strong>{len(dropped_cols)}</strong> column(s) (<em>{', '.join(dropped_cols[:4])}{'...' if len(dropped_cols)>4 else ''}</em>) due to >80% missing or constant values.")
        if label_mappings:
            insights.append(f"<strong>{len(label_mappings)}</strong> categorical column(s) were label-encoded: {', '.join(list(label_mappings.keys())[:5])}.")
        if problem_type == 'classification':
            if extra_metrics.get('F1 Score', 100) < 70:
                insights.append("⚠ F1 Score is below 70% — consider gathering more data or trying feature engineering.")
        else:
            if best['score'] < 50:
                insights.append("⚠ R² Score is below 50% — the target may have low linear predictability. Try adding more features.")
        n_test = len(y_test)
        insights.append(f"Train/test split: <strong>{len(X_train)}</strong> training rows · <strong>{n_test}</strong> test rows (80/20 split).")
    except Exception:
        pass

    # ── Persist model + metadata ───────────────────────────────────────────────
    os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    meta_path  = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')

    if best_model is not None:
        joblib.dump({
            'model':          best_model,
            'scaler':         scaler,
            'encoders':       encoders,
            'label_mappings': label_mappings,
            'feature_names':  feature_names,
            'problem_type':   problem_type,
            'target':         target,
        }, model_path)

    session_results = [{'name': r['name'], 'score': r['score']} for r in results]

    metadata = {
        'problem_type':    problem_type,
        'target':          target,
        'metric_label':    metric_label,
        'feature_names':   feature_names,
        'label_mappings':  label_mappings,
        'best_model_name': best['name'],
        'best_score':      best['score'],
        'extra_metrics':   extra_metrics,
        'preprocessing': {
            'original_rows':    original_rows,
            'original_cols':    original_cols,
            'duplicates_count': duplicates_count,
            'rows_after_clean': rows_after,
            'rows_dropped':     rows_before - rows_after,
            'cols_dropped':     dropped_cols,
            'cols_after':       len(feature_names) + 1,
            'encoded_cols':     list(label_mappings.keys()),
            'n_train':          len(X_train),
            'n_test':           len(y_test),
            'null_info':        null_info,
            'imputation_log':   imputation_log,
        },
    }
    with open(meta_path, 'w') as fp:
        json.dump(metadata, fp)

    request.session['ml_results']         = session_results
    request.session['ml_metric_label']    = metric_label
    request.session['best_model_name']    = best['name']
    request.session['problem_type']       = problem_type
    request.session['feature_importance'] = feature_importance
    request.session['feature_names']      = feature_names
    request.session['label_mappings']     = label_mappings
    request.session['ml_charts']          = charts_data
    request.session['ml_extra_metrics']   = extra_metrics
    request.session['ml_insights']        = insights
    request.session['ml_preprocessing']   = metadata['preprocessing']

    return redirect('results')


def results(request):
    ml_results = request.session.get('ml_results')
    if not ml_results:
        messages.error(request, 'No results yet. Please train a model first.')
        return redirect('upload')

    context = {
        'results':          ml_results,
        'metric_label':     request.session.get('ml_metric_label', 'Score'),
        'best_model':       request.session.get('best_model_name', ''),
        'problem_type':     request.session.get('problem_type', ''),
        'feature_importance': request.session.get('feature_importance', []),
        'target':           request.session.get('target_column', ''),
        'dataset_name':     request.session.get('dataset_name', 'dataset'),
        'charts':           request.session.get('ml_charts', {}),
        'extra_metrics':    request.session.get('ml_extra_metrics', {}),
        'insights':         request.session.get('ml_insights', []),
        'preprocessing':    request.session.get('ml_preprocessing', {}),
        'label_mappings':   request.session.get('label_mappings', {}),
    }
    return render(request, 'result.html', context)


def report_pdf(request):
    """Generate a university-level PDF report of the ML pipeline."""
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        messages.error(request, 'No trained model found. Train a model first.')
        return redirect('results')

    with open(meta_path) as fp:
        meta = json.load(fp)

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage,
                                     HRFlowable, PageBreak, KeepTogether)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    import re

    W, H = A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title='AutoAnalyst AI — ML Report',
        author='AutoAnalyst AI',
    )

    # ── Paragraph styles ──────────────────────────────────────────────────────
    BLACK   = colors.HexColor('#09090b')
    DARK    = colors.HexColor('#3f3f46')
    MUTED   = colors.HexColor('#71717a')
    MUTED2  = colors.HexColor('#a1a1aa')
    GREEN   = colors.HexColor('#16a34a')
    BORDER  = colors.HexColor('#e4e4e7')
    BG      = colors.HexColor('#fafafa')
    WHITE   = colors.white

    sH1  = ParagraphStyle('H1',  fontSize=22, leading=26, textColor=BLACK, fontName='Helvetica-Bold', spaceAfter=4)
    sH2  = ParagraphStyle('H2',  fontSize=14, leading=18, textColor=BLACK, fontName='Helvetica-Bold', spaceBefore=18, spaceAfter=6)
    sH3  = ParagraphStyle('H3',  fontSize=11, leading=14, textColor=DARK,  fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4)
    sH4  = ParagraphStyle('H4',  fontSize=10, leading=13, textColor=DARK,  fontName='Helvetica-Bold', spaceBefore=6,  spaceAfter=3)
    sBod = ParagraphStyle('Bod', fontSize=9,  leading=14, textColor=DARK,  fontName='Helvetica', spaceAfter=4)
    sMut = ParagraphStyle('Mut', fontSize=8,  leading=11, textColor=MUTED, fontName='Helvetica')
    sCen = ParagraphStyle('Cen', fontSize=9,  leading=12, textColor=DARK,  fontName='Helvetica', alignment=TA_CENTER)
    sRig = ParagraphStyle('Rig', fontSize=9,  leading=12, textColor=DARK,  fontName='Helvetica', alignment=TA_RIGHT)
    sBold= ParagraphStyle('Bold',fontSize=9,  leading=13, textColor=BLACK, fontName='Helvetica-Bold', spaceAfter=3)

    def table_style(header_color=BLACK, alt=True):
        ts = [
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
        ]
        if alt:
            ts += [('ROWBACKGROUNDS', (0,1), (-1,-1), [BG, WHITE])]
        return TableStyle(ts)

    def hr():
        return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=4)

    def section_spacer():
        return Spacer(1, 10)

    def add_chart(story, b64_str, caption='', width_cm=13):
        img_bytes = io.BytesIO(base64.b64decode(b64_str))
        img = RLImage(img_bytes, width=width_cm*cm, height=width_cm*0.62*cm)
        story.append(img)
        if caption:
            story.append(Paragraph(caption, sMut))
        story.append(Spacer(1, 6))

    def clean_html(text):
        return re.sub(r'<[^>]+>', '', str(text))

    story = []
    pre  = meta.get('preprocessing', {})
    now  = datetime.datetime.now().strftime('%B %d, %Y')
    dataset_name = request.session.get('dataset_name', 'N/A')

    # ════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph('AutoAnalyst AI', ParagraphStyle(
        'Cover', fontSize=28, leading=32, textColor=BLACK, fontName='Helvetica-Bold',
        alignment=TA_CENTER)))
    story.append(Spacer(1, 6))
    story.append(Paragraph('Machine Learning Pipeline Report', ParagraphStyle(
        'CoverSub', fontSize=14, leading=18, textColor=MUTED, fontName='Helvetica',
        alignment=TA_CENTER)))
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width='60%', thickness=1.5, color=BLACK, hAlign='CENTER'))
    story.append(Spacer(1, 0.8*cm))

    # Meta table
    meta_cover = [
        ['Dataset',       dataset_name],
        ['Target Column', meta.get('target', 'N/A')],
        ['Problem Type',  meta.get('problem_type', 'N/A').title()],
        ['Best Model',    meta.get('best_model_name', 'N/A')],
        [meta.get('metric_label', 'Score'), f"{meta.get('best_score', 'N/A')}%"],
        ['Report Date',   now],
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
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        'Generated by AutoAnalyst AI — Automated Machine Learning Platform',
        ParagraphStyle('FootCover', fontSize=8, textColor=MUTED2,
                       fontName='Helvetica', alignment=TA_CENTER)
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 1 — DATASET OVERVIEW
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('1. Dataset Overview', sH2))
    story.append(hr())
    story.append(Paragraph(
        f'The dataset <strong>{dataset_name}</strong> was uploaded and processed through the '
        f'AutoAnalyst pipeline. The table below summarizes its key structural properties.',
        sBod))
    story.append(section_spacer())

    ov_data = [
        ['Property', 'Value', 'Notes'],
        ['Original Rows',     str(pre.get('original_rows', 'N/A')),
         'Total records in the uploaded file'],
        ['Original Columns',  str(pre.get('original_cols', 'N/A')),
         'Total features including target'],
        ['Duplicate Rows',    str(pre.get('duplicates_count', 'N/A')),
         'Identical rows detected in the dataset'],
        ['Rows After Cleaning', str(pre.get('rows_after_clean', 'N/A')),
         'Rows remaining after null-target removal'],
        ['Rows Dropped',      str(pre.get('rows_dropped', 'N/A')),
         'Rows removed due to missing target value'],
        ['Columns Used',      str(pre.get('cols_after', 'N/A')),
         'Columns used after dropping uninformative ones'],
        ['Training Samples',  str(pre.get('n_train', 'N/A')),
         '80% of cleaned data (random split, seed=42)'],
        ['Test Samples',      str(pre.get('n_test',  'N/A')),
         '20% of cleaned data (held-out evaluation set)'],
    ]
    t_ov = Table(ov_data, colWidths=[4.5*cm, 3.5*cm, 8*cm])
    t_ov.setStyle(table_style())
    story.append(t_ov)

    # Dropped columns
    dropped = pre.get('cols_dropped', [])
    if dropped:
        story.append(section_spacer())
        story.append(Paragraph('Columns Removed During Cleaning', sH3))
        story.append(Paragraph(
            'The following columns were automatically removed because they exceeded '
            '80% missing values, had zero variance, or were identified as non-informative '
            'identifiers (e.g. ID, email, name):', sBod))
        dc_data = [['#', 'Column Name', 'Removal Reason']]
        for i, col in enumerate(dropped, 1):
            dc_data.append([str(i), col, 'High missing rate / constant / identifier column'])
        t_dc = Table(dc_data, colWidths=[1*cm, 7*cm, 8*cm])
        t_dc.setStyle(table_style())
        story.append(t_dc)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 2 — DATA QUALITY & MISSING VALUE ANALYSIS
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('2. Data Quality & Missing Value Analysis', sH2))
    story.append(hr())
    null_info = pre.get('null_info', {})

    if null_info:
        story.append(Paragraph(
            f'The dataset contained missing values in <strong>{len(null_info)}</strong> column(s). '
            f'Missing numeric values were filled using the column <strong>median</strong> '
            f'(robust to outliers), while missing categorical values were filled using the '
            f'column <strong>mode</strong> (most frequent value). '
            f'This strategy ensures no information is lost and the model trains on a complete dataset.',
            sBod))
        story.append(section_spacer())

        mv_data = [['Column', 'Data Type', 'Missing Count', 'Missing %', 'Imputation Strategy']]
        for col, info in null_info.items():
            dtype_str = info.get('dtype', 'N/A')
            if 'int' in dtype_str or 'float' in dtype_str:
                strategy = 'Median imputation'
            else:
                strategy = 'Mode imputation'
            mv_data.append([
                col,
                dtype_str,
                str(info.get('count', 0)),
                f"{info.get('pct', 0)}%",
                strategy,
            ])
        t_mv = Table(mv_data, colWidths=[4*cm, 3*cm, 2.5*cm, 2.5*cm, 4*cm])
        t_mv.setStyle(table_style())

        # Highlight high missing % rows in yellow
        ts_extra = list(t_mv._tblStyle._cmds)
        for row_i, (col, info) in enumerate(null_info.items(), 1):
            if info.get('pct', 0) > 30:
                ts_extra.append(('BACKGROUND', (0, row_i), (-1, row_i),
                                  colors.HexColor('#fefce8')))
        story.append(t_mv)

        # Imputation detail
        imputation_log = pre.get('imputation_log', [])
        if imputation_log:
            story.append(section_spacer())
            story.append(Paragraph('Imputation Detail', sH3))
            il_data = [['Column', 'Missing Count', 'Strategy', 'Fill Value']]
            for item in imputation_log:
                il_data.append([
                    item.get('column', ''),
                    str(item.get('missing', '')),
                    item.get('strategy', ''),
                    str(item.get('fill_value', '')),
                ])
            t_il = Table(il_data, colWidths=[4.5*cm, 3*cm, 4*cm, 4.5*cm])
            t_il.setStyle(table_style())
            story.append(t_il)
    else:
        story.append(Paragraph(
            '✓ No missing values were detected in this dataset. All columns contained complete data.',
            ParagraphStyle('Ok', fontSize=10, leading=14, textColor=GREEN, fontName='Helvetica-Bold')))

    # Duplicates
    story.append(section_spacer())
    story.append(Paragraph('Duplicate Row Handling', sH3))
    dup_count = pre.get('duplicates_count', 0)
    if dup_count > 0:
        story.append(Paragraph(
            f'The dataset contained <strong>{dup_count}</strong> duplicate row(s). '
            f'Note: AutoAnalyst focuses on rows with null target values for removal; '
            f'exact duplicates are retained to preserve data volume unless they represent '
            f'copy errors. You may drop them manually using <em>df.drop_duplicates()</em> if needed.',
            sBod))
    else:
        story.append(Paragraph(
            '✓ No duplicate rows were found in the dataset.',
            ParagraphStyle('Ok', fontSize=10, leading=14, textColor=GREEN, fontName='Helvetica-Bold')))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 3 — FEATURE ENGINEERING & ENCODING
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('3. Feature Engineering & Encoding', sH2))
    story.append(hr())
    lm = meta.get('label_mappings', {})
    feature_names = meta.get('feature_names', [])

    story.append(Paragraph(
        f'The pipeline used <strong>{len(feature_names)}</strong> feature(s) to predict '
        f'<strong>{meta.get("target", "target")}</strong>. '
        f'All numeric features were standardized using <strong>StandardScaler</strong> '
        f'(zero mean, unit variance). '
        f'Categorical (text) features were converted to numbers using '
        f'<strong>Label Encoding</strong> before standardization.',
        sBod))
    story.append(section_spacer())

    story.append(Paragraph('Feature List', sH3))
    feat_data = [['#', 'Feature Name', 'Type', 'Encoding Applied']]
    for i, feat in enumerate(feature_names, 1):
        if feat in lm:
            feat_type = 'Categorical'
            enc = 'Label Encoding + StandardScaler'
        else:
            feat_type = 'Numeric'
            enc = 'StandardScaler only'
        feat_data.append([str(i), feat, feat_type, enc])
    t_feat = Table(feat_data, colWidths=[1*cm, 6*cm, 3*cm, 6*cm])
    t_feat.setStyle(table_style())
    story.append(t_feat)

    if lm:
        story.append(section_spacer())
        story.append(Paragraph('Label Encoding Mappings', sH3))
        story.append(Paragraph(
            'Each unique string value in a categorical column was assigned a unique integer. '
            'During prediction, the same mapping is applied to user input. '
            'The table below shows the original string values and their encoded integers.',
            sBod))
        enc_data = [['Column', 'Original Values → Encoded Integers']]
        for col, vals in lm.items():
            mapping_str = '  |  '.join([f'{v} → {i}' for i, v in enumerate(vals[:12])])
            if len(vals) > 12:
                mapping_str += f'  | ... ({len(vals)-12} more)'
            enc_data.append([col, mapping_str])
        te = Table(enc_data, colWidths=[4*cm, 12*cm])
        te.setStyle(table_style())
        story.append(te)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 4 — MODEL TRAINING & SELECTION
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('4. Model Training & Selection', sH2))
    story.append(hr())
    problem_type = meta.get('problem_type', 'classification')
    story.append(Paragraph(
        f'Problem type detected: <strong>{problem_type.title()}</strong>. '
        f'The pipeline automatically trained multiple algorithms and selected the '
        f'best-performing model based on <strong>{meta.get("metric_label", "Score")}</strong> '
        f'on a held-out 20% test set. The train/test split used a fixed random seed (42) '
        f'for reproducibility.',
        sBod))
    story.append(section_spacer())

    if problem_type == 'classification':
        story.append(Paragraph(
            'Algorithms Evaluated: Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine.',
            sBod))
    else:
        story.append(Paragraph(
            'Algorithms Evaluated: Linear Regression, Random Forest Regressor, Ridge Regression.',
            sBod))

    ml_results = request.session.get('ml_results', [])
    if ml_results:
        story.append(section_spacer())
        r_data = [['Rank', 'Model', f"{meta['metric_label']} (%)", 'Status']]
        for i, r in enumerate(sorted(ml_results, key=lambda x: x['score'], reverse=True), 1):
            is_best = r['name'] == meta['best_model_name']
            r_data.append([
                str(i), r['name'], f"{r['score']}%",
                '★ Best Model' if is_best else '—'
            ])
        tr = Table(r_data, colWidths=[1.5*cm, 8*cm, 4*cm, 3*cm])
        ts = table_style(header_color=BLACK)
        # Highlight best row
        for row_i, r in enumerate(sorted(ml_results, key=lambda x: x['score'], reverse=True), 1):
            if r['name'] == meta['best_model_name']:
                ts._cmds.append(('BACKGROUND', (0, row_i), (-1, row_i),
                                  colors.HexColor('#f0fdf4')))
                ts._cmds.append(('TEXTCOLOR', (0, row_i), (-1, row_i), GREEN))
                ts._cmds.append(('FONTNAME',  (0, row_i), (-1, row_i), 'Helvetica-Bold'))
        tr.setStyle(ts)
        story.append(tr)

    # Extra metrics
    extra = meta.get('extra_metrics', {})
    if extra:
        story.append(section_spacer())
        story.append(Paragraph('Additional Evaluation Metrics (Best Model)', sH3))
        em_data = [['Metric', 'Value', 'Description']]
        metric_desc = {
            'Precision': 'Fraction of positive predictions that are actually correct',
            'Recall':    'Fraction of actual positives correctly identified',
            'F1 Score':  'Harmonic mean of Precision and Recall',
            'MAE':       'Mean Absolute Error — average prediction error magnitude',
            'RMSE':      'Root Mean Squared Error — penalizes large errors more',
            'MSE':       'Mean Squared Error — average squared prediction error',
        }
        for k, v in extra.items():
            unit = '%' if isinstance(v, float) and v > 1 else ''
            em_data.append([k, f'{v}{unit}', metric_desc.get(k, '')])
        tem = Table(em_data, colWidths=[3.5*cm, 3*cm, 9.5*cm])
        tem.setStyle(table_style(header_color=colors.HexColor('#3f3f46')))
        story.append(tem)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 5 — VISUAL ANALYSIS
    # ════════════════════════════════════════════════════════════
    charts_data = request.session.get('ml_charts', {})
    if charts_data:
        story.append(Paragraph('5. Visual Analysis', sH2))
        story.append(hr())

        if 'model_comparison' in charts_data:
            story.append(Paragraph('5.1 Model Comparison', sH3))
            story.append(Paragraph(
                f'The bar chart below compares the {meta.get("metric_label","Score")} of all '
                f'trained models. The best-performing model is highlighted in green.', sBod))
            add_chart(story, charts_data['model_comparison'],
                      f'Figure 1: Model comparison by {meta.get("metric_label","Score")}')

        if 'feature_importance' in charts_data:
            story.append(section_spacer())
            story.append(Paragraph('5.2 Feature Importance', sH3))
            story.append(Paragraph(
                'Feature importance shows which input variables had the greatest influence '
                'on the model\'s predictions. Higher bars indicate stronger predictive power.', sBod))
            add_chart(story, charts_data['feature_importance'],
                      'Figure 2: Feature importance scores from the best model')

        if 'confusion' in charts_data:
            story.append(section_spacer())
            story.append(Paragraph('5.3 Confusion Matrix', sH3))
            story.append(Paragraph(
                'The confusion matrix shows how many samples were correctly and incorrectly '
                'classified per class. Diagonal cells represent correct predictions; '
                'off-diagonal cells represent misclassifications.', sBod))
            add_chart(story, charts_data['confusion'],
                      'Figure 3: Confusion matrix on the 20% test set', width_cm=9)

        if 'actual_vs_pred' in charts_data:
            story.append(section_spacer())
            story.append(Paragraph('5.3 Actual vs Predicted', sH3))
            story.append(Paragraph(
                'Each point represents one test sample. Points closer to the red dashed '
                'diagonal line indicate more accurate predictions. Scatter indicates model error.', sBod))
            add_chart(story, charts_data['actual_vs_pred'],
                      'Figure 3: Actual vs Predicted values on the 20% test set')

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 6 — FEATURE IMPORTANCE TABLE
    # ════════════════════════════════════════════════════════════
    fi = request.session.get('feature_importance', [])
    if fi:
        story.append(Paragraph('6. Feature Importance Rankings', sH2))
        story.append(hr())
        story.append(Paragraph(
            'The table below ranks each feature by its contribution to the model\'s '
            'decision-making process. For tree-based models this is based on Gini impurity '
            'reduction; for linear models it is based on the absolute coefficient value.',
            sBod))
        story.append(section_spacer())
        fi_data = [['Rank', 'Feature Name', 'Importance Score', 'Relative Strength']]
        max_imp = fi[0]['importance'] if fi else 1
        for i, item in enumerate(fi, 1):
            pct = round(item['importance'] / max_imp * 100, 1)
            bars = '█' * int(pct / 10)
            fi_data.append([str(i), item['feature'], f"{item['importance']:.4f}", f"{bars} {pct}%"])
        tfi = Table(fi_data, colWidths=[1.5*cm, 7*cm, 4*cm, 4.5*cm])
        tfi.setStyle(table_style())
        story.append(tfi)
        story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # SECTION 7 — AUTO-GENERATED INSIGHTS
    # ════════════════════════════════════════════════════════════
    insights = request.session.get('ml_insights', [])
    sec_num = 7 if fi else 6
    story.append(Paragraph(f'{sec_num}. Auto-Generated Insights & Recommendations', sH2))
    story.append(hr())
    story.append(Paragraph(
        'The following observations were automatically generated by analyzing '
        'the dataset structure, preprocessing results, and model performance.',
        sBod))
    story.append(section_spacer())
    if insights:
        for ins in insights:
            clean = clean_html(ins)
            story.append(Paragraph(f'• {clean}', sBod))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph('No insights generated.', sMut))

    story.append(section_spacer())
    story.append(Paragraph('Recommendations', sH3))
    recs = [
        'Collect more data if model accuracy is below expectations — especially for rare classes.',
        'Apply feature selection techniques (e.g., Recursive Feature Elimination) to reduce noise.',
        'Consider hyperparameter tuning (GridSearchCV) to further improve the best model.',
        'Use cross-validation (k=5 or k=10) for a more robust performance estimate.',
        'If the dataset has class imbalance, consider SMOTE or class_weight adjustments.',
    ]
    for rec in recs:
        story.append(Paragraph(f'→ {rec}', sBod))
        story.append(Spacer(1, 2))

    # ════════════════════════════════════════════════════════════
    # SECTION 8 — PIPELINE SUMMARY
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph(f'{sec_num+1}. Pipeline Execution Summary', sH2))
    story.append(hr())
    pipeline_steps = [
        ['Step', 'Action', 'Outcome'],
        ['1. Data Ingestion',
         f'Loaded {dataset_name}',
         f'{pre.get("original_rows","N/A")} rows × {pre.get("original_cols","N/A")} columns'],
        ['2. Null Target Removal',
         'Removed rows where target is null',
         f'{pre.get("rows_dropped","0")} rows removed'],
        ['3. Column Pruning',
         'Dropped high-missing / constant / ID columns',
         f'{len(dropped)} column(s) removed' if dropped else 'No columns removed'],
        ['4. Missing Value Imputation',
         'Numeric → Median, Categorical → Mode',
         f'{len(pre.get("null_info",{}))} column(s) imputed'],
        ['5. Label Encoding',
         'LabelEncoder on categorical features',
         f'{len(meta.get("label_mappings",{}))} column(s) encoded'],
        ['6. Feature Scaling',
         'StandardScaler (mean=0, std=1)',
         f'{len(feature_names)} feature(s) scaled'],
        ['7. Train/Test Split',
         '80% train / 20% test, random_state=42',
         f'{pre.get("n_train","N/A")} train / {pre.get("n_test","N/A")} test'],
        ['8. Model Training',
         f'Trained {len(ml_results) if ml_results else "N/A"} model(s)',
         f'Best: {meta.get("best_model_name","N/A")}'],
        ['9. Evaluation',
         f'{meta.get("metric_label","Score")} on test set',
         f'{meta.get("best_score","N/A")}%'],
        ['10. Model Persistence',
         'Saved best model as best_model.joblib',
         'Ready for prediction'],
    ]
    t_pipe = Table(pipeline_steps, colWidths=[4*cm, 6*cm, 6*cm])
    t_pipe.setStyle(table_style())
    story.append(t_pipe)

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f'AutoAnalyst AI  ·  Report generated on {now}  ·  All rights reserved.',
        ParagraphStyle('Footer', fontSize=7.5, textColor=MUTED2,
                       fontName='Helvetica', alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    response = HttpResponse(buf.read(), content_type='application/pdf')
    base_name = dataset_name.rsplit('.', 1)[0]
    response['Content-Disposition'] = (
        f'attachment; filename="AutoAnalyst_{base_name}_Report_{datetime.date.today()}.pdf"'
    )
    return response
