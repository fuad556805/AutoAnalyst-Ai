import os
import json
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import FileResponse, Http404
import joblib


def predict(request):
    meta_path  = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        messages.error(request, 'No trained model found. Please train a model first.')
        return redirect('select_target')

    with open(meta_path) as fp:
        meta = json.load(fp)

    feature_names    = meta['feature_names']
    problem_type     = meta['problem_type']
    best_model_name  = meta['best_model_name']
    metric_label     = meta['metric_label']
    best_score       = meta['best_score']
    label_mappings   = meta.get('label_mappings', {})   # {col: [val0, val1, ...]}

    prediction_result = None
    error_msg         = None

    if request.method == 'POST':
        try:
            loaded   = joblib.load(model_path)
            model    = loaded['model']
            scaler   = loaded['scaler']
            encoders = loaded.get('encoders', {})

            input_vals = []
            for feat in feature_names:
                raw = request.POST.get(feat, '').strip()

                if feat in encoders:
                    # Categorical feature — user selected original string value
                    if raw == '' or raw is None:
                        raw = label_mappings.get(feat, [''])[0]
                    try:
                        val = float(encoders[feat].transform([str(raw)])[0])
                    except Exception:
                        val = 0.0
                else:
                    # Numeric feature
                    try:
                        val = float(raw) if raw != '' else 0.0
                    except ValueError:
                        val = 0.0

                input_vals.append(val)

            X_input  = np.array(input_vals).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            pred     = model.predict(X_scaled)[0]

            # For classification, reverse-decode if target was encoded
            target_col = meta.get('target', '')
            target_lm  = label_mappings.get(target_col)
            if problem_type == 'classification' and target_lm is not None:
                try:
                    pred_label = target_lm[int(pred)]
                except Exception:
                    pred_label = str(pred)
            else:
                pred_label = str(pred)

            if problem_type == 'classification':
                prediction_result = {
                    'value':       pred_label,
                    'label':       'Predicted Class',
                    'explanation': (
                        f'The model classified this input as <strong>{pred_label}</strong> '
                        f'using {best_model_name} ({metric_label}: {best_score}%).'
                    ),
                }
            else:
                pred_rounded = round(float(pred), 4)
                prediction_result = {
                    'value':       f'{pred_rounded:,}',
                    'label':       'Predicted Value',
                    'explanation': (
                        f'The model estimated <strong>{target_col}</strong> as '
                        f'<strong>{pred_rounded}</strong> using {best_model_name} '
                        f'({metric_label}: {best_score}%).'
                    ),
                }
        except Exception as e:
            error_msg = str(e)

    context = {
        'feature_names':   feature_names,
        'problem_type':    problem_type,
        'best_model_name': best_model_name,
        'metric_label':    metric_label,
        'best_score':      best_score,
        'label_mappings':  label_mappings,
        'prediction_result': prediction_result,
        'error_msg':       error_msg,
        'target':          meta.get('target', ''),
    }
    return render(request, 'predict.html', context)


def download_model(request):
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        raise Http404('Model not found')
    return FileResponse(open(model_path, 'rb'), as_attachment=True,
                        filename='best_model.joblib')
