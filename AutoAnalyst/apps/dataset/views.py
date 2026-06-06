import os
import json
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


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
            if ext == 'csv':
                df = pd.read_csv(save_path)
            else:
                df = pd.read_excel(save_path)
        except Exception as e:
            messages.error(request, f'Could not read file: {e}')
            return redirect('upload')

        request.session['dataset_path'] = save_path
        request.session['dataset_name'] = f.name
        request.session.pop('target_column', None)
        request.session.pop('ml_results', None)
        request.session.pop('best_model_name', None)

        return redirect('preview')

    return render(request, 'upload.html')


def preview(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    ext = path.rsplit('.', 1)[1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    # Basic profiling
    shape = df.shape
    columns = list(df.columns)
    dtypes = {col: str(df[col].dtype) for col in columns}
    missing = df.isnull().sum().to_dict()
    missing_pct = {col: round(v / shape[0] * 100, 2) for col, v in missing.items()}
    duplicates = int(df.duplicated().sum())
    unique_counts = {col: int(df[col].nunique()) for col in columns}
    preview_rows = df.head(10).fillna('').to_dict(orient='records')

    # Suggest target column (last column or column named 'target'/'label'/'class')
    suggested_target = columns[-1]
    for col in columns:
        if col.lower() in ('target', 'label', 'class', 'output', 'y', 'result'):
            suggested_target = col
            break

    # Column stats
    col_stats = []
    for col in columns:
        stat = {
            'name': col,
            'dtype': dtypes[col],
            'missing': missing[col],
            'missing_pct': missing_pct[col],
            'unique': unique_counts[col],
        }
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            stat['min'] = round(float(df[col].min()), 4) if not pd.isna(df[col].min()) else 'N/A'
            stat['max'] = round(float(df[col].max()), 4) if not pd.isna(df[col].max()) else 'N/A'
            stat['mean'] = round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else 'N/A'
        else:
            stat['min'] = stat['max'] = stat['mean'] = 'N/A'
        col_stats.append(stat)

    context = {
        'dataset_name': request.session.get('dataset_name', 'dataset'),
        'rows': shape[0],
        'cols': shape[1],
        'duplicates': duplicates,
        'preview_rows': preview_rows,
        'columns': columns,
        'col_stats': col_stats,
        'suggested_target': suggested_target,
    }
    return render(request, 'preview.html', context)


def select_target(request):
    path = request.session.get('dataset_path')
    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')

    ext = path.rsplit('.', 1)[1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
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
        'columns': columns,
        'suggested_target': suggested_target,
        'dataset_name': request.session.get('dataset_name', 'dataset'),
    }
    return render(request, 'select_target.html', context)
