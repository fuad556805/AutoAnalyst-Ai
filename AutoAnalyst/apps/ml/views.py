import os
import io
import json
import base64
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
    label_mappings = {}   # {col: [original_val0, val1, ...]} index = encoded number

    # Step 1 — Impute BEFORE encoding
    for col in df.columns:
        if df[col].isna().any():
            if _is_numeric(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill)

    # Step 2 — Encode all non-numeric columns; record original classes
    for col in df.columns:
        if not _is_numeric(df[col]):
            le = LabelEncoder()
            original_classes = sorted(df[col].astype(str).unique().tolist())
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            label_mappings[col] = list(le.classes_)  # in encoded order

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

    return X_train, X_test, y_train, y_test, scaler, encoders, label_mappings, feature_names


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
        df, rows_before, rows_after, dropped_cols = _clean_df(df, target)
        problem_type = _detect_problem_type(df[target])

        X_train, X_test, y_train, y_test, scaler, encoders, label_mappings, feature_names = \
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

        # Confusion matrix for best model
        if best.get('preds') is not None:
            try:
                labels = sorted(y_test.unique())
                charts_data['confusion'] = _chart_confusion(
                    best['y_test'], best['preds'], labels=labels)
            except Exception:
                pass

        # Classification metrics
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

        # Actual vs Predicted chart
        if best.get('preds') is not None:
            try:
                charts_data['actual_vs_pred'] = _chart_actual_vs_pred(
                    best['y_test'], best['preds'])
            except Exception:
                pass

        # Regression metrics
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

    # Model comparison chart
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
            'original_rows':  original_rows,
            'original_cols':  original_cols,
            'rows_after_clean': rows_after,
            'rows_dropped':   rows_before - rows_after,
            'cols_dropped':   dropped_cols,
            'cols_after':     len(feature_names) + 1,
            'encoded_cols':   list(label_mappings.keys()),
            'n_train':        len(X_train),
            'n_test':         len(y_test),
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
    """Generate and download a PDF report of the ML results."""
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
                                     HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle('h1', fontSize=20, leading=24, textColor=colors.HexColor('#09090b'),
                                spaceAfter=6, fontName='Helvetica-Bold')
    style_h2 = ParagraphStyle('h2', fontSize=13, leading=16, textColor=colors.HexColor('#09090b'),
                                spaceAfter=4, spaceBefore=14, fontName='Helvetica-Bold')
    style_h3 = ParagraphStyle('h3', fontSize=11, leading=13, textColor=colors.HexColor('#3f3f46'),
                                spaceAfter=3, spaceBefore=8, fontName='Helvetica-Bold')
    style_body = ParagraphStyle('body', fontSize=9, leading=13, textColor=colors.HexColor('#3f3f46'),
                                 spaceAfter=3, fontName='Helvetica')
    style_muted = ParagraphStyle('muted', fontSize=8, leading=11, textColor=colors.HexColor('#71717a'),
                                  fontName='Helvetica')
    style_center = ParagraphStyle('center', fontSize=9, leading=12, alignment=TA_CENTER,
                                   textColor=colors.HexColor('#3f3f46'), fontName='Helvetica')

    story = []

    # ── Title ──────────────────────────────────────────────────────────────────
    story.append(Paragraph('AutoAnalyst AI — ML Report', style_h1))
    story.append(Paragraph(
        f"Dataset: {request.session.get('dataset_name', 'N/A')}  ·  Target: {meta['target']}  ·  Problem: {meta['problem_type']}",
        style_muted))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e4e4e7'), spaceAfter=10))

    # ── Preprocessing ──────────────────────────────────────────────────────────
    pre = meta.get('preprocessing', {})
    story.append(Paragraph('1. Data Preprocessing', style_h2))
    pre_data = [
        ['Metric', 'Value'],
        ['Original rows',      str(pre.get('original_rows', 'N/A'))],
        ['Rows after cleaning', str(pre.get('rows_after_clean', 'N/A'))],
        ['Rows dropped',       str(pre.get('rows_dropped', 'N/A'))],
        ['Original columns',   str(pre.get('original_cols', 'N/A'))],
        ['Columns used',       str(pre.get('cols_after', 'N/A'))],
        ['Encoded columns',    ', '.join(pre.get('encoded_cols', [])) or 'None'],
        ['Training samples',   str(pre.get('n_train', 'N/A'))],
        ['Test samples',       str(pre.get('n_test', 'N/A'))],
    ]
    t = Table(pre_data, colWidths=[7*cm, 9*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#09090b')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e4e4e7')),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING',(0,0), (-1,-1), 8),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]))
    story.append(t)

    # ── Label Encoding ────────────────────────────────────────────────────────
    lm = meta.get('label_mappings', {})
    if lm:
        story.append(Paragraph('2. Label Encoding', style_h2))
        story.append(Paragraph('Original string values → numeric labels used in training:', style_body))
        enc_data = [['Column', 'Original Values → Encoded']]
        for col, vals in lm.items():
            mapping_str = ', '.join([f'{v}→{i}' for i, v in enumerate(vals)])
            enc_data.append([col, mapping_str])
        te = Table(enc_data, colWidths=[6*cm, 10*cm])
        te.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#09090b')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e4e4e7')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING',(0,0), (-1,-1), 8),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ]))
        story.append(te)

    # ── Model Results ─────────────────────────────────────────────────────────
    story.append(Paragraph('3. Model Performance', style_h2))
    story.append(Paragraph(f"Metric: {meta['metric_label']}  ·  Best model: {meta['best_model_name']}  ·  Score: {meta['best_score']}%", style_body))
    ml_results = request.session.get('ml_results', [])
    if ml_results:
        r_data = [['Rank', 'Model', f"{meta['metric_label']} (%)","Status"]]
        for i, r in enumerate(sorted(ml_results, key=lambda x: x['score'], reverse=True), 1):
            r_data.append([str(i), r['name'], str(r['score']),
                           'Best ✓' if r['name'] == meta['best_model_name'] else ''])
        tr = Table(r_data, colWidths=[1.5*cm, 8*cm, 4*cm, 3*cm])
        tr.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#09090b')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e4e4e7')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING',(0,0), (-1,-1), 8),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ]))
        story.append(tr)

    # Extra metrics
    extra = meta.get('extra_metrics', {})
    if extra:
        story.append(Spacer(1, 8))
        em_data = [['Metric', 'Value']] + [[k, str(v)] for k, v in extra.items()]
        tem = Table(em_data, colWidths=[7*cm, 9*cm])
        tem.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3f3f46')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e4e4e7')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING',(0,0), (-1,-1), 8),
            ('TOPPADDING',  (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ]))
        story.append(tem)

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_data = request.session.get('ml_charts', {})
    if charts_data:
        story.append(Paragraph('4. Visual Analysis', style_h2))

        def add_chart(b64_str, width_cm=14):
            img_bytes = io.BytesIO(base64.b64decode(b64_str))
            img = RLImage(img_bytes, width=width_cm*cm, height=width_cm*0.7*cm)
            story.append(img)
            story.append(Spacer(1, 6))

        if 'model_comparison' in charts_data:
            story.append(Paragraph('Model Comparison', style_h3))
            add_chart(charts_data['model_comparison'])
        if 'feature_importance' in charts_data:
            story.append(Paragraph('Feature Importance', style_h3))
            add_chart(charts_data['feature_importance'])
        if 'confusion' in charts_data:
            story.append(Paragraph('Confusion Matrix', style_h3))
            add_chart(charts_data['confusion'], width_cm=9)
        if 'actual_vs_pred' in charts_data:
            story.append(Paragraph('Actual vs Predicted', style_h3))
            add_chart(charts_data['actual_vs_pred'])

    # ── Feature Importance table ───────────────────────────────────────────────
    fi = request.session.get('feature_importance', [])
    if fi:
        story.append(Paragraph('5. Feature Importance Rankings', style_h2))
        fi_data = [['Rank', 'Feature', 'Importance']]
        for i, item in enumerate(fi, 1):
            fi_data.append([str(i), item['feature'], str(item['importance'])])
        tfi = Table(fi_data, colWidths=[1.5*cm, 11*cm, 4*cm])
        tfi.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#09090b')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e4e4e7')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING',(0,0), (-1,-1), 8),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ]))
        story.append(tfi)

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = request.session.get('ml_insights', [])
    if insights:
        story.append(Paragraph('6. Auto-Generated Insights', style_h2))
        for ins in insights:
            # Strip HTML tags for PDF
            import re
            clean = re.sub(r'<[^>]+>', '', ins)
            story.append(Paragraph(f'• {clean}', style_body))
            story.append(Spacer(1, 2))

    doc.build(story)
    buf.seek(0)
    response = HttpResponse(buf.read(), content_type='application/pdf')
    dataset_name = request.session.get('dataset_name', 'report').rsplit('.', 1)[0]
    response['Content-Disposition'] = f'attachment; filename="autoanalyst_{dataset_name}_report.pdf"'
    return response
