import os
import io
import base64
import json
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.views.decorators.http import require_POST

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Helpers ────────────────────────────────────────────────────────────────────

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


def _read_df(path):
    ext = path.rsplit('.', 1)[1].lower()
    return pd.read_csv(path) if ext == 'csv' else pd.read_excel(path)


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


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


# ── Views ──────────────────────────────────────────────────────────────────────

def upload(request):
    if request.method == 'POST':
        if 'dataset' not in request.FILES:
            messages.error(request, 'No file selected.')
            return redirect('upload')

        f = request.FILES['dataset']
        if not _allowed_file(f.name):
            messages.error(request, 'Only CSV and XLSX files are supported.')
            return redirect('upload')

        ext = f.name.rsplit('.', 1)[1].lower()
        save_path = os.path.join(settings.MEDIA_ROOT, f.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(save_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        try:
            df = _read_df(save_path)
        except Exception as e:
            messages.error(request, f'Could not read file: {e}')
            return redirect('upload')

        # Clear old ML session state
        for key in ['target_column', 'ml_results', 'best_model_name', 'problem_type',
                    'feature_importance', 'feature_names', 'label_mappings',
                    'ml_charts', 'ml_extra_metrics', 'ml_insights', 'ml_preprocessing']:
            request.session.pop(key, None)

        request.session['dataset_path'] = save_path
        request.session['dataset_name'] = f.name
        return redirect('preview')

    current_dataset = request.session.get('dataset_name')
    current_path    = request.session.get('dataset_path')
    dataset_exists  = bool(current_path and os.path.exists(current_path))

    return render(request, 'upload.html', {
        'current_dataset': current_dataset if dataset_exists else None,
    })


@require_POST
def delete_dataset(request):
    path = request.session.get('dataset_path')
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    for key in ['dataset_path', 'dataset_name', 'target_column', 'ml_results',
                'best_model_name', 'problem_type', 'feature_importance', 'feature_names',
                'label_mappings', 'ml_charts', 'ml_extra_metrics', 'ml_insights',
                'ml_preprocessing', 'ml_metric_label']:
        request.session.pop(key, None)
    messages.success(request, 'Dataset deleted. You can upload a new file.')
    return redirect('upload')


def preview(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = _read_df(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    shape    = df.shape
    columns  = list(df.columns)
    dtypes   = {col: str(df[col].dtype) for col in columns}
    missing  = df.isnull().sum().to_dict()
    missing_pct = {col: round(v / max(shape[0], 1) * 100, 2) for col, v in missing.items()}
    duplicates  = int(df.duplicated().sum())
    unique_counts = {col: int(df[col].nunique()) for col in columns}
    preview_rows  = df.head(10).fillna('').to_dict(orient='records')

    suggested_target = columns[-1]
    for col in columns:
        if col.lower() in ('target', 'label', 'class', 'output', 'y', 'result', 'survived'):
            suggested_target = col
            break

    col_stats = []
    for col in columns:
        stat = {
            'name':        col,
            'dtype':       dtypes[col],
            'missing':     missing[col],
            'missing_pct': missing_pct[col],
            'unique':      unique_counts[col],
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            stat['min']  = round(float(df[col].min()), 4)  if not pd.isna(df[col].min())  else 'N/A'
            stat['max']  = round(float(df[col].max()), 4)  if not pd.isna(df[col].max())  else 'N/A'
            stat['mean'] = round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else 'N/A'
        else:
            stat['min'] = stat['max'] = stat['mean'] = 'N/A'
            stat['top'] = str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A'
        col_stats.append(stat)

    context = {
        'dataset_name':   request.session.get('dataset_name', 'dataset'),
        'rows':           shape[0],
        'cols':           shape[1],
        'duplicates':     duplicates,
        'preview_rows':   preview_rows,
        'columns':        columns,
        'col_stats':      col_stats,
        'suggested_target': suggested_target,
        'missing_total':  sum(missing.values()),
    }
    return render(request, 'preview.html', context)


def select_target(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = _read_df(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    columns = list(df.columns)

    if request.method == 'POST':
        target = request.POST.get('target_column')
        if target not in columns:
            messages.error(request, 'Invalid target column.')
            return redirect('select_target')
        request.session['target_column'] = target
        return redirect('train')

    suggested_target = request.session.get('target_column', columns[-1])
    context = {
        'columns':          columns,
        'suggested_target': suggested_target,
        'dataset_name':     request.session.get('dataset_name', 'dataset'),
    }
    return render(request, 'select_target.html', context)


def visualize(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = _read_df(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    target = request.session.get('target_column')
    charts = []

    def make_chart(title, chart_type, b64):
        charts.append({'title': title, 'type': chart_type, 'img': b64})

    # ── Numeric distributions ─────────────────────────────────────────────────
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols[:8]:
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            data = df[col].dropna()
            ax.hist(data, bins=min(30, max(10, len(data)//20)),
                    color='#18181b', edgecolor='white', linewidth=0.5, alpha=0.88)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        make_chart(col, 'histogram', _fig_b64(fig))

    # ── Categorical bar charts ────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(12)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            bar_colors = ['#16a34a' if v == target else '#18181b' for v in vc.index]
            ax.bar(vc.index.astype(str), vc.values,
                   color='#18181b', edgecolor='none', width=0.6, alpha=0.88)
            ax.set_title(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=40)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        make_chart(col, 'bar', _fig_b64(fig))

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        heat_cols = num_cols[:12]
        corr = df[heat_cols].corr()
        n = len(heat_cols)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(max(5, n * 0.75), max(4, n * 0.7)))
            im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(heat_cols, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(heat_cols, fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = corr.values[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(val) > 0.6 else '#09090b')
            ax.set_title('Feature Correlation Matrix')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        make_chart('Correlation Matrix', 'heatmap', _fig_b64(fig))

    # ── Target distribution ───────────────────────────────────────────────────
    if target and target in df.columns:
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            if pd.api.types.is_numeric_dtype(df[target]):
                ax.hist(df[target].dropna(), bins=20, color='#16a34a',
                        edgecolor='white', linewidth=0.5, alpha=0.88)
                ax.set_xlabel(target)
                ax.set_ylabel('Count')
            else:
                vc = df[target].value_counts()
                ax.bar(vc.index.astype(str), vc.values, color='#16a34a',
                       edgecolor='none', width=0.6, alpha=0.88)
                ax.tick_params(axis='x', rotation=40)
            ax.set_title(f'Target: {target}')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        make_chart(f'Target — {target}', 'target', _fig_b64(fig))

    context = {
        'charts':       charts,
        'dataset_name': request.session.get('dataset_name', ''),
        'target':       target,
        'n_charts':     len(charts),
    }
    return render(request, 'visualize.html', context)
