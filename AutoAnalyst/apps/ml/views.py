import os
import json
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def _detect_problem_type(series):
    """Classification if target is categorical OR has ≤ 20 unique values."""
    if not _is_numeric(series):
        return 'classification'
    n_unique = series.nunique()
    if n_unique <= 20:
        return 'classification'
    return 'regression'


def _clean_df(df, target_col):
    """Drop useless columns; keep target no matter what."""
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        missing_pct = df[col].isna().mean()
        # Drop if >80% missing
        if missing_pct > 0.80:
            cols_to_drop.append(col)
            continue
        # Drop if constant
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
            continue
        # Drop ID-like columns by name
        if col.lower() in ('id', 'index', 'row', 'no', 'num', '#', 'uuid',
                            'email', 'phone', 'name', 'rowid'):
            cols_to_drop.append(col)
            continue
        # Drop high-cardinality pure-string cols (>90% unique, not target)
        if not _is_numeric(df[col]) and df[col].nunique() / max(len(df), 1) > 0.90:
            cols_to_drop.append(col)
            continue

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df.copy()


def _preprocess(df, target_col):
    """
    Full preprocessing pipeline:
      1. Impute missing values (mode for strings, median for numerics)
      2. Encode all non-numeric columns with LabelEncoder
      3. Final coerce — any value still non-numeric → 0
      4. Scale features with StandardScaler
    Returns: X_train, X_test, y_train, y_test, scaler, encoders, feature_names
    """
    encoders = {}

    # Step 1 — Impute BEFORE encoding
    for col in df.columns:
        if df[col].isna().any():
            if _is_numeric(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill)

    # Step 2 — Encode ALL non-numeric columns (object, category, bool, etc.)
    for col in df.columns:
        if not _is_numeric(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Step 3 — Final safety: coerce any remaining non-numeric to 0
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Step 4 — Split X / y
    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col]

    feature_names = list(X.columns)

    # Step 5 — Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, scaler, encoders, feature_names


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
        messages.error(request, f'Target column "{target}" not found. Please re-select.')
        return redirect('select_target')

    if len(df) < 5:
        messages.error(request, 'Dataset is too small (need at least 5 rows).')
        return redirect('upload')

    try:
        # Clean
        df = _clean_df(df, target)

        # Detect problem type BEFORE encoding (while target may still be string)
        problem_type = _detect_problem_type(df[target])

        # Preprocess
        X_train, X_test, y_train, y_test, scaler, encoders, feature_names = \
            _preprocess(df.copy(), target)

    except Exception as e:
        messages.error(request, f'Preprocessing error: {e}')
        return redirect('select_target')

    # ── Train models ──────────────────────────────────────────────────────────
    results = []

    if problem_type == 'classification':
        metric_label = 'Accuracy'
        model_list = [
            ('Logistic Regression',   LogisticRegression(max_iter=2000, random_state=42)),
            ('Random Forest',         RandomForestClassifier(n_estimators=100, random_state=42)),
            ('K-Nearest Neighbors',   KNeighborsClassifier(n_neighbors=min(5, len(X_train)))),
            ('Support Vector Machine',SVC(random_state=42, probability=False)),
        ]
        for name, model in model_list:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = round(accuracy_score(y_test, preds) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None})
    else:
        metric_label = 'R² Score'
        model_list = [
            ('Linear Regression',        LinearRegression()),
            ('Random Forest Regressor',  RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Ridge Regression',         Ridge()),
        ]
        for name, model in model_list:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                raw_r2 = r2_score(y_test, preds)
                score  = round(max(raw_r2, 0) * 100, 2)
                results.append({'name': name, 'score': score, 'model': model})
            except Exception as e:
                results.append({'name': name, 'score': 0.0, 'model': None})

    # Sort descending by score
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # ── Feature importance ────────────────────────────────────────────────────
    feature_importance = []
    best_model = best['model']
    if best_model is not None:
        try:
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                pairs = sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True)[:10]
                feature_importance = [
                    {'feature': f, 'importance': round(float(v), 4)} for f, v in pairs
                ]
            elif hasattr(best_model, 'coef_'):
                coef = np.array(best_model.coef_).flatten()
                pairs = sorted(zip(feature_names, np.abs(coef)),
                               key=lambda x: x[1], reverse=True)[:10]
                feature_importance = [
                    {'feature': f, 'importance': round(float(v), 4)} for f, v in pairs
                ]
        except Exception:
            feature_importance = []

    # ── Persist model ─────────────────────────────────────────────────────────
    os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    meta_path  = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')

    if best_model is not None:
        joblib.dump({
            'model':         best_model,
            'scaler':        scaler,
            'encoders':      encoders,
            'feature_names': feature_names,
            'problem_type':  problem_type,
            'target':        target,
        }, model_path)

    metadata = {
        'problem_type':   problem_type,
        'target':         target,
        'metric_label':   metric_label,
        'feature_names':  feature_names,
        'best_model_name':best['name'],
        'best_score':     best['score'],
    }
    with open(meta_path, 'w') as fp:
        json.dump(metadata, fp)

    # ── Store in session ──────────────────────────────────────────────────────
    request.session['ml_results']       = [{'name': r['name'], 'score': r['score']} for r in results]
    request.session['ml_metric_label']  = metric_label
    request.session['best_model_name']  = best['name']
    request.session['problem_type']     = problem_type
    request.session['feature_importance'] = feature_importance
    request.session['feature_names']    = feature_names

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
    }
    return render(request, 'result.html', context)
