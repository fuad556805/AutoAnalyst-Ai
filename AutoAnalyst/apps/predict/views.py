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
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        messages.error(request, 'No trained model found. Please train a model first.')
        return redirect('train')

    with open(meta_path) as fp:
        meta = json.load(fp)

    feature_names = meta['feature_names']
    problem_type = meta['problem_type']
    best_model_name = meta['best_model_name']
    metric_label = meta['metric_label']
    best_score = meta['best_score']

    prediction_result = None
    error_msg = None

    if request.method == 'POST':
        try:
            loaded = joblib.load(model_path)
            model = loaded['model']
            scaler = loaded['scaler']
            encoders = loaded['encoders']

            input_vals = []
            for feat in feature_names:
                val = request.POST.get(feat, '')
                if val == '':
                    val = 0.0
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
                input_vals.append(val)

            X_input = np.array(input_vals).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)[0]

            if problem_type == 'classification':
                prediction_result = {
                    'value': str(pred),
                    'label': 'Predicted Class',
                    'explanation': f'The model classified this input as <strong>{pred}</strong> using {best_model_name}.',
                }
            else:
                prediction_result = {
                    'value': f'{round(float(pred), 4)}',
                    'label': 'Predicted Value',
                    'explanation': f'The model estimated the value as <strong>{round(float(pred), 4)}</strong> using {best_model_name}.',
                }
        except Exception as e:
            error_msg = str(e)

    context = {
        'feature_names': feature_names,
        'problem_type': problem_type,
        'best_model_name': best_model_name,
        'metric_label': metric_label,
        'best_score': best_score,
        'prediction_result': prediction_result,
        'error_msg': error_msg,
    }
    return render(request, 'predict.html', context)


def download_model(request):
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        raise Http404("Model not found")
    return FileResponse(open(model_path, 'rb'), as_attachment=True, filename='best_model.joblib')
