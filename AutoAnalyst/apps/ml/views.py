import os
import json
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse

from .utils.preprocessing import clean_df, preprocess, detect_problem_type
from .utils.training import run_classification, run_regression, generate_insights
from .utils.report import generate_pdf_report

import joblib


# ── Train ──────────────────────────────────────────────────────────────────────

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

        null_info = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                null_info[col] = {
                    'count': n,
                    'pct': round(n / len(df) * 100, 2),
                    'dtype': str(df[col].dtype),
                }

        df, rows_before, rows_after, dropped_cols = clean_df(df, target)
        problem_type = detect_problem_type(df[target])

        (X_train, X_test, y_train, y_test,
         scaler, encoders, label_mappings,
         feature_names, imputation_log) = preprocess(df.copy(), target)
    except Exception as e:
        messages.error(request, f'Preprocessing error: {e}')
        return redirect('select_target')

    # ── Run training ───────────────────────────────────────────────────────────
    try:
        if problem_type == 'classification':
            output = run_classification(X_train, X_test, y_train, y_test, feature_names)
        else:
            output = run_regression(X_train, X_test, y_train, y_test, feature_names)
    except Exception as e:
        messages.error(request, f'Training error: {e}')
        return redirect('select_target')

    results          = output['results']
    best             = output['best']
    charts_data      = output['charts_data']
    extra_metrics    = output['extra_metrics']
    feature_importance = output['feature_importance']
    metric_label     = output['metric_label']
    best_model       = best['model']

    # ── Semantic labels for numeric target classes ──────────────────────────────
    # When the target column is numeric (e.g. 0/1), LabelEncoder doesn't encode
    # it, so label_mappings has no entry. Build a meaningful name from the column.
    if problem_type == 'classification' and target not in label_mappings:
        try:
            classes = list(best_model.classes_)
            int_classes = sorted([int(c) for c in classes])
            col_name = target.replace('_', ' ').replace('-', ' ').strip().title()
            if int_classes == [0, 1]:
                # Binary 0/1: 0 → "Not <Column>", 1 → "<Column>"
                label_mappings[target] = [f'Not {col_name}', col_name]
            else:
                # Multi-class numeric: label as "Class 0", "Class 1", …
                label_mappings[target] = [
                    f'Class {c}' for c in sorted(int_classes)
                ]
        except Exception:
            pass

    insights = generate_insights(
        best=best,
        results=results,
        dropped_cols=dropped_cols,
        label_mappings=label_mappings,
        problem_type=problem_type,
        extra_metrics=extra_metrics,
        n_train=len(X_train),
        n_test=len(y_test),
        metric_label=metric_label,
    )

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
    preprocessing_meta = {
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
    }
    metadata = {
        'problem_type':    problem_type,
        'target':          target,
        'metric_label':    metric_label,
        'feature_names':   feature_names,
        'label_mappings':  label_mappings,
        'best_model_name': best['name'],
        'best_score':      best['score'],
        'extra_metrics':   extra_metrics,
        'preprocessing':   preprocessing_meta,
    }
    with open(meta_path, 'w') as fp:
        json.dump(metadata, fp)

    # ── Save TrainingRun to DB ─────────────────────────────────────────────────
    training_run_id = None
    if request.user.is_authenticated:
        from apps.dashboard.models import TrainingRun, Dataset as DatasetRecord
        dataset_id = request.session.get('current_dataset_id')
        dataset_obj = None
        if dataset_id:
            try:
                dataset_obj = DatasetRecord.objects.get(pk=dataset_id, user=request.user)
            except DatasetRecord.DoesNotExist:
                pass

        run = TrainingRun.objects.create(
            user=request.user,
            dataset=dataset_obj,
            dataset_name=request.session.get('dataset_name', ''),
            target_column=target,
            problem_type=problem_type,
            best_model_name=best['name'],
            best_score=best['score'],
            metric_label=metric_label,
            n_features=len(feature_names),
            n_train=len(X_train),
            n_test=len(y_test),
            all_results=session_results,
        )
        training_run_id = run.pk

    # ── Session ────────────────────────────────────────────────────────────────
    request.session['ml_results']          = session_results
    request.session['ml_metric_label']     = metric_label
    request.session['best_model_name']     = best['name']
    request.session['problem_type']        = problem_type
    request.session['feature_importance']  = feature_importance
    request.session['feature_names']       = feature_names
    request.session['label_mappings']      = label_mappings
    request.session['ml_charts']           = charts_data
    request.session['ml_extra_metrics']    = extra_metrics
    request.session['ml_insights']         = insights
    request.session['ml_preprocessing']    = preprocessing_meta
    if training_run_id:
        request.session['current_training_run_id'] = training_run_id

    return redirect('results')


# ── Results ────────────────────────────────────────────────────────────────────

def results(request):
    ml_results = request.session.get('ml_results')
    if not ml_results:
        messages.error(request, 'No results yet. Please train a model first.')
        return redirect('upload')

    context = {
        'results':            ml_results,
        'metric_label':       request.session.get('ml_metric_label', 'Score'),
        'best_model':         request.session.get('best_model_name', ''),
        'problem_type':       request.session.get('problem_type', ''),
        'feature_importance': request.session.get('feature_importance', []),
        'target':             request.session.get('target_column', ''),
        'dataset_name':       request.session.get('dataset_name', 'dataset'),
        'charts':             request.session.get('ml_charts', {}),
        'extra_metrics':      request.session.get('ml_extra_metrics', {}),
        'insights':           request.session.get('ml_insights', []),
        'preprocessing':      request.session.get('ml_preprocessing', {}),
        'label_mappings':     request.session.get('label_mappings', {}),
    }
    return render(request, 'result.html', context)


# ── PDF Report ─────────────────────────────────────────────────────────────────

def report_pdf(request):
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        messages.error(request, 'No trained model found. Train a model first.')
        return redirect('results')

    with open(meta_path) as fp:
        meta = json.load(fp)

    pdf_bytes = generate_pdf_report(
        meta=meta,
        dataset_name=request.session.get('dataset_name', 'dataset'),
        ml_results=request.session.get('ml_results', []),
        charts_data=request.session.get('ml_charts', {}),
        feature_importance=request.session.get('feature_importance', []),
        insights=request.session.get('ml_insights', []),
    )

    dataset_name = request.session.get('dataset_name', 'dataset').rsplit('.', 1)[0]
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="AutoAnalyst_Report_{dataset_name}.pdf"'
    return response
