import os
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.views.decorators.http import require_POST

from .utils.file_handler import allowed_file, read_df, get_file_size
from .utils.charts import build_visualizations


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload(request):
    if request.method == 'POST':
        if 'dataset' not in request.FILES:
            messages.error(request, 'No file selected.')
            return redirect('upload')

        f = request.FILES['dataset']
        if not allowed_file(f.name):
            messages.error(request, 'Only CSV and XLSX files are supported.')
            return redirect('upload')

        save_path = os.path.join(settings.MEDIA_ROOT, f.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(save_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        try:
            df = read_df(save_path)
        except Exception as e:
            messages.error(request, f'Could not read file: {e}')
            return redirect('upload')

        for key in ['target_column', 'ml_results', 'best_model_name', 'problem_type',
                    'feature_importance', 'feature_names', 'label_mappings',
                    'ml_charts', 'ml_extra_metrics', 'ml_insights', 'ml_preprocessing',
                    'current_dataset_id']:
            request.session.pop(key, None)

        request.session['dataset_path'] = save_path
        request.session['dataset_name'] = f.name

        # ── Persist to DB (authenticated users only) ───────────────────────────
        if request.user.is_authenticated:
            from apps.dashboard.models import Dataset as DatasetRecord
            rec = DatasetRecord.objects.create(
                user=request.user,
                name=f.name,
                file_path=save_path,
                file_size=get_file_size(save_path),
                rows=df.shape[0],
                columns=df.shape[1],
            )
            request.session['current_dataset_id'] = rec.pk

        return redirect('preview')

    current_dataset = request.session.get('dataset_name')
    current_path    = request.session.get('dataset_path')
    dataset_exists  = bool(current_path and os.path.exists(current_path))

    return render(request, 'upload.html', {
        'current_dataset': current_dataset if dataset_exists else None,
    })


# ── Delete ─────────────────────────────────────────────────────────────────────

@require_POST
def delete_dataset(request):
    path = request.session.get('dataset_path')
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    if request.user.is_authenticated:
        dataset_id = request.session.get('current_dataset_id')
        if dataset_id:
            from apps.dashboard.models import Dataset as DatasetRecord
            DatasetRecord.objects.filter(pk=dataset_id, user=request.user).delete()

    for key in ['dataset_path', 'dataset_name', 'target_column', 'ml_results',
                'best_model_name', 'problem_type', 'feature_importance', 'feature_names',
                'label_mappings', 'ml_charts', 'ml_extra_metrics', 'ml_insights',
                'ml_preprocessing', 'ml_metric_label', 'current_dataset_id']:
        request.session.pop(key, None)

    messages.success(request, 'Dataset deleted. You can upload a new file.')
    return redirect('upload')


# ── Preview ────────────────────────────────────────────────────────────────────

def preview(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = read_df(path)
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

    # ── After-cleaning estimates (before target is selected) ──────────────────
    rows_after_dedup   = shape[0] - duplicates
    high_missing_cols  = [c for c in columns if missing_pct[c] > 80]
    constant_cols      = [c for c in columns if unique_counts[c] <= 1]
    id_like_cols       = [c for c in columns
                          if c.lower() in ('id', 'uuid', 'index', 'email', 'phone')]
    dropped_cols_est   = list({*high_missing_cols, *constant_cols, *id_like_cols})
    cols_after_est     = max(shape[1] - len(dropped_cols_est), 1)
    missing_total      = sum(missing.values())

    context = {
        'dataset_name':      request.session.get('dataset_name', 'dataset'),
        'rows':              shape[0],
        'cols':              shape[1],
        'duplicates':        duplicates,
        'preview_rows':      preview_rows,
        'columns':           columns,
        'col_stats':         col_stats,
        'suggested_target':  suggested_target,
        'missing_total':     missing_total,
        # after-cleaning estimates
        'rows_after_dedup':  rows_after_dedup,
        'cols_after_est':    cols_after_est,
        'dropped_cols_est':  dropped_cols_est,
        'high_missing_cols': high_missing_cols,
        'constant_cols':     constant_cols,
    }
    return render(request, 'preview.html', context)


# ── Select Target ──────────────────────────────────────────────────────────────

def select_target(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = read_df(path)
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

    context = {
        'columns':          columns,
        'suggested_target': request.session.get('target_column', columns[-1]),
        'dataset_name':     request.session.get('dataset_name', 'dataset'),
    }
    return render(request, 'select_target.html', context)


# ── Visualize ──────────────────────────────────────────────────────────────────

def visualize(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    try:
        df = read_df(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    target = request.session.get('target_column')
    charts = build_visualizations(df, target)

    context = {
        'charts':       charts,
        'dataset_name': request.session.get('dataset_name', ''),
        'target':       target,
        'n_charts':     len(charts),
    }
    return render(request, 'visualize.html', context)
