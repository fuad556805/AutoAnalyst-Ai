import os
import json
import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib


def _detect_problem_type(series):
    n_unique = series.nunique()
    if series.dtype == 'object' or n_unique <= 10:
        return 'classification'
    return 'regression'


def _clean_df(df, target_col):
    # Drop columns with >60% missing
    thresh = int(len(df) * 0.4)
    df = df.dropna(axis=1, thresh=thresh)
    # Keep target col even if it has missings
    df = df.dropna(subset=[target_col])
    # Drop constant columns (except target)
    for col in df.columns:
        if col != target_col and df[col].nunique() <= 1:
            df = df.drop(columns=[col])
    # Drop ID-like columns
    for col in df.columns:
        if col != target_col and col.lower() in ('id', 'index', 'row', 'no', 'num', '#'):
            df = df.drop(columns=[col])
    return df


def train(request):
    path = request.session.get('dataset_path')
    target = request.session.get('target_column')

    if not path or not os.path.exists(path):
        messages.error(request, 'No dataset loaded. Please upload first.')
        return redirect('upload')
    if not target:
        messages.error(request, 'Please select a target column first.')
        return redirect('select_target')

    ext = path.rsplit('.', 1)[1].lower()
    try:
        df = pd.read_csv(path) if ext == 'csv' else pd.read_excel(path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('upload')

    if target not in df.columns:
        messages.error(request, 'Target column not found in dataset.')
        return redirect('select_target')

    df = _clean_df(df, target)
    problem_type = _detect_problem_type(df[target])

    # Encode categoricals
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Fill missing with median
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target])
    y = df[target]

    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = []

    if problem_type == 'classification':
        models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('K-Nearest Neighbors', KNeighborsClassifier()),
            ('Support Vector Machine', SVC(random_state=42)),
        ]
        metric_label = 'Accuracy'
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = round(accuracy_score(y_test, preds) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None, 'error': str(e)})
    else:
        models = [
            ('Linear Regression', LinearRegression()),
            ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Ridge Regression', Ridge()),
        ]
        metric_label = 'R² Score'
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = round(r2_score(y_test, preds) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None, 'error': str(e)})

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # Feature importance
    feature_importance = []
    best_model = best['model']
    if best_model is not None:
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            feature_importance = [{'feature': f, 'importance': round(float(v), 4)} for f, v in pairs]
        elif hasattr(best_model, 'coef_'):
            coef = best_model.coef_
            if hasattr(coef, 'flatten'):
                coef = coef.flatten()
            pairs = sorted(zip(feature_names, abs(coef)), key=lambda x: x[1], reverse=True)[:10]
            feature_importance = [{'feature': f, 'importance': round(float(v), 4)} for f, v in pairs]

    # Save best model + metadata
    os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')

    if best_model is not None:
        joblib.dump({'model': best_model, 'scaler': scaler, 'encoders': encoders,
                     'feature_names': feature_names, 'problem_type': problem_type,
                     'target': target}, model_path)

    metadata = {
        'problem_type': problem_type,
        'target': target,
        'metric_label': metric_label,
        'feature_names': feature_names,
        'best_model_name': best['name'],
        'best_score': best['score'],
    }
    with open(meta_path, 'w') as fp:
        json.dump(metadata, fp)

    # Serialize results for session (remove model objects)
    session_results = [{'name': r['name'], 'score': r['score']} for r in results]

    request.session['ml_results'] = session_results
    request.session['ml_metric_label'] = metric_label
    request.session['best_model_name'] = best['name']
    request.session['problem_type'] = problem_type
    request.session['feature_importance'] = feature_importance
    request.session['feature_names'] = feature_names

    return redirect('results')


def results(request):
    ml_results = request.session.get('ml_results')
    if not ml_results:
        messages.error(request, 'No results yet. Please train a model first.')
        return redirect('upload')

    context = {
        'results': ml_results,
        'metric_label': request.session.get('ml_metric_label', 'Score'),
        'best_model': request.session.get('best_model_name', ''),
        'problem_type': request.session.get('problem_type', ''),
        'feature_importance': request.session.get('feature_importance', []),
        'target': request.session.get('target_column', ''),
        'dataset_name': request.session.get('dataset_name', 'dataset'),
    }
    return render(request, 'result.html', context)
