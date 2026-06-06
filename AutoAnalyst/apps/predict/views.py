import os
import io
import json
import textwrap
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
import joblib


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(request):
    meta_path  = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        messages.error(request, 'No trained model found. Please train a model first.')
        return redirect('select_target')

    with open(meta_path) as fp:
        meta = json.load(fp)

    feature_names   = meta['feature_names']
    problem_type    = meta['problem_type']
    best_model_name = meta['best_model_name']
    metric_label    = meta['metric_label']
    best_score      = meta['best_score']
    label_mappings  = meta.get('label_mappings', {})

    prediction_result = None
    error_msg         = None

    if request.method == 'POST':
        try:
            loaded   = joblib.load(model_path)
            model    = loaded['model']
            scaler   = loaded['scaler']
            encoders = loaded.get('encoders', {})

            input_vals    = []
            input_summary = {}

            for feat in feature_names:
                raw = request.POST.get(feat, '').strip()
                if feat in encoders:
                    if not raw:
                        raw = label_mappings.get(feat, [''])[0]
                    try:
                        val = float(encoders[feat].transform([str(raw)])[0])
                    except Exception:
                        val = 0.0
                    input_summary[feat] = raw
                else:
                    try:
                        val = float(raw) if raw else 0.0
                    except ValueError:
                        val = 0.0
                    input_summary[feat] = raw if raw else '0'
                input_vals.append(val)

            X_input  = np.array(input_vals).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            pred     = model.predict(X_scaled)[0]

            target_col = meta.get('target', '')
            target_lm  = label_mappings.get(target_col)

            if problem_type == 'classification':
                if target_lm is not None:
                    try:
                        pred_label = target_lm[int(pred)]
                    except Exception:
                        pred_label = str(pred)
                else:
                    try:
                        fval = float(pred)
                        pred_label = str(int(fval)) if fval == int(fval) else str(pred)
                    except Exception:
                        pred_label = str(pred)

                try:
                    if target_lm is not None:
                        all_classes = [str(c) for c in target_lm]
                    else:
                        raw_classes = list(model.classes_)
                        all_classes = []
                        for c in raw_classes:
                            try:
                                fval = float(c)
                                all_classes.append(str(int(fval)) if fval == int(fval) else str(c))
                            except Exception:
                                all_classes.append(str(c))
                    classes_str = ', '.join(f'<em>{c}</em>' for c in all_classes)
                except Exception:
                    classes_str = ''

                explanation = (
                    f'The model <strong>{best_model_name}</strong> analysed the '
                    f'{len(feature_names)} input feature(s) and classified '
                    f'<strong>{target_col}</strong> as <strong>"{pred_label}"</strong>. '
                )
                if classes_str:
                    explanation += f'Possible classes: {classes_str}. '
                explanation += f'Model {metric_label}: <strong>{best_score}%</strong>.'

                prediction_result = {
                    'value':         pred_label,
                    'label':         f'Predicted: {target_col}',
                    'input_summary': input_summary,
                    'explanation':   explanation,
                }

                # ── Save Prediction to DB ──────────────────────────────────────
                _save_prediction(request, best_model_name, target_col,
                                 pred_label, input_summary)

            else:
                pred_rounded = round(float(pred), 4)
                if pred_rounded == int(pred_rounded):
                    formatted = f'{int(pred_rounded):,}'
                else:
                    formatted = f'{pred_rounded:,.4f}'.rstrip('0').rstrip('.')

                prediction_result = {
                    'value':       formatted,
                    'label':       f'Predicted {target_col}',
                    'input_summary': input_summary,
                    'explanation': (
                        f'<strong>{best_model_name}</strong> estimated '
                        f'<strong>{target_col}</strong> = <strong>{formatted}</strong> '
                        f'from {len(feature_names)} feature(s). '
                        f'R² Score: <strong>{best_score}%</strong>.'
                    ),
                }

                # ── Save Prediction to DB ──────────────────────────────────────
                _save_prediction(request, best_model_name, target_col,
                                 formatted, input_summary)

        except Exception as e:
            error_msg = str(e)

    context = {
        'feature_names':     feature_names,
        'problem_type':      problem_type,
        'best_model_name':   best_model_name,
        'metric_label':      metric_label,
        'best_score':        best_score,
        'label_mappings':    label_mappings,
        'prediction_result': prediction_result,
        'error_msg':         error_msg,
        'target':            meta.get('target', ''),
    }
    return render(request, 'predict.html', context)


def _save_prediction(request, model_name: str, target_col: str,
                     result_value: str, input_summary: dict):
    if not request.user.is_authenticated:
        return
    try:
        from apps.dashboard.models import Prediction, TrainingRun
        run_id = request.session.get('current_training_run_id')
        run_obj = None
        if run_id:
            try:
                run_obj = TrainingRun.objects.get(pk=run_id, user=request.user)
            except TrainingRun.DoesNotExist:
                pass
        Prediction.objects.create(
            user=request.user,
            training_run=run_obj,
            model_name=model_name,
            target_column=target_col,
            result_value=result_value,
            input_summary=input_summary,
        )
    except Exception:
        pass


# ── Download Model ─────────────────────────────────────────────────────────────

def download_model(request):
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        raise Http404('Model not found')
    return FileResponse(open(model_path, 'rb'), as_attachment=True,
                        filename='best_model.joblib')


# ── Download Notebook ──────────────────────────────────────────────────────────

def download_notebook(request):
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        messages.error(request, 'No trained model found. Train a model first.')
        return redirect('predict')

    with open(meta_path) as fp:
        meta = json.load(fp)

    from .utils.notebook import generate_notebook
    nb_json = generate_notebook(meta, request.session.get('dataset_name', 'dataset.csv'))

    base_name = request.session.get('dataset_name', 'dataset').rsplit('.', 1)[0]
    response = HttpResponse(nb_json, content_type='application/x-ipynb+json')
    response['Content-Disposition'] = f'attachment; filename="AutoAnalyst_{base_name}.ipynb"'
    return response
