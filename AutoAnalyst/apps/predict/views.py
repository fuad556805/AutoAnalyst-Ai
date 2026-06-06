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
    label_mappings   = meta.get('label_mappings', {})

    prediction_result = None
    error_msg         = None

    if request.method == 'POST':
        try:
            loaded   = joblib.load(model_path)
            model    = loaded['model']
            scaler   = loaded['scaler']
            encoders = loaded.get('encoders', {})

            input_vals = []
            input_summary = {}

            for feat in feature_names:
                raw = request.POST.get(feat, '').strip()

                if feat in encoders:
                    if raw == '' or raw is None:
                        raw = label_mappings.get(feat, [''])[0]
                    try:
                        val = float(encoders[feat].transform([str(raw)])[0])
                    except Exception:
                        val = 0.0
                    input_summary[feat] = raw
                else:
                    try:
                        val = float(raw) if raw != '' else 0.0
                    except ValueError:
                        val = 0.0
                    input_summary[feat] = raw if raw != '' else '0'

                input_vals.append(val)

            X_input  = np.array(input_vals).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            pred     = model.predict(X_scaled)[0]

            target_col = meta.get('target', '')
            target_lm  = label_mappings.get(target_col)

            if problem_type == 'classification':
                # Case 1 — target was string-encoded (label_mappings holds the decoder)
                if target_lm is not None:
                    try:
                        pred_label = target_lm[int(pred)]
                    except Exception:
                        pred_label = str(pred)
                else:
                    # Case 2 — target was already numeric; model.predict() returns
                    # the actual class value directly (e.g. 0, 1, 2, …)
                    # Just convert cleanly: 0.0 → "0", 1.0 → "1"
                    try:
                        fval = float(pred)
                        if fval == int(fval):
                            pred_label = str(int(fval))
                        else:
                            pred_label = str(pred)
                    except Exception:
                        pred_label = str(pred)

                # Build a human-readable label line
                display_label = f'Predicted: {target_col}'

                # Collect all possible class labels for context
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
                    f'<strong>{target_col}</strong> as '
                    f'<strong>"{pred_label}"</strong>. '
                )
                if classes_str:
                    explanation += f'Possible classes: {classes_str}. '
                explanation += (
                    f'Model {metric_label} on the test set: '
                    f'<strong>{best_score}%</strong>.'
                )

                prediction_result = {
                    'value':         pred_label,
                    'label':         display_label,
                    'input_summary': input_summary,
                    'explanation':   explanation,
                }

            else:
                pred_rounded = round(float(pred), 4)
                # Format: remove trailing zeros for whole numbers
                if pred_rounded == int(pred_rounded):
                    formatted = f'{int(pred_rounded):,}'
                else:
                    formatted = f'{pred_rounded:,.4f}'.rstrip('0').rstrip('.')

                prediction_result = {
                    'value':         formatted,
                    'label':         f'Predicted {target_col}',
                    'input_summary': input_summary,
                    'explanation': (
                        f'<strong>{best_model_name}</strong> estimated the value of '
                        f'<strong>{target_col}</strong> to be '
                        f'<strong>{formatted}</strong> based on the '
                        f'{len(feature_names)} input feature(s). '
                        f'Model R² Score on the test set: '
                        f'<strong>{best_score}%</strong>.'
                    ),
                }
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


def download_model(request):
    model_path = os.path.join(settings.SAVED_MODELS_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        raise Http404('Model not found')
    return FileResponse(open(model_path, 'rb'), as_attachment=True,
                        filename='best_model.joblib')


def download_notebook(request):
    """Generate and download a Jupyter Notebook (.ipynb) for the trained pipeline."""
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        messages.error(request, 'No trained model found. Train a model first.')
        return redirect('predict')

    with open(meta_path) as fp:
        meta = json.load(fp)

    problem_type    = meta.get('problem_type', 'classification')
    target          = meta.get('target', 'target')
    best_model_name = meta.get('best_model_name', 'Random Forest')
    metric_label    = meta.get('metric_label', 'Accuracy')
    best_score      = meta.get('best_score', 0)
    feature_names   = meta.get('feature_names', [])
    label_mappings  = meta.get('label_mappings', {})
    pre             = meta.get('preprocessing', {})
    encoded_cols    = pre.get('encoded_cols', [])
    dataset_name    = request.session.get('dataset_name', 'dataset.csv')

    # ── Model class mapping ────────────────────────────────
    clf_map = {
        'Logistic Regression':    'LogisticRegression(max_iter=2000, random_state=42)',
        'Random Forest':          'RandomForestClassifier(n_estimators=100, random_state=42)',
        'K-Nearest Neighbors':    'KNeighborsClassifier(n_neighbors=5)',
        'Support Vector Machine': 'SVC(random_state=42)',
    }
    reg_map = {
        'Linear Regression':       'LinearRegression()',
        'Random Forest Regressor': 'RandomForestRegressor(n_estimators=100, random_state=42)',
        'Ridge Regression':        'Ridge()',
    }
    model_init = (clf_map if problem_type == 'classification' else reg_map).get(
        best_model_name, 'RandomForestClassifier(n_estimators=100, random_state=42)'
    )

    imports_clf = textwrap.dedent("""\
        from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns""")

    imports_reg = textwrap.dedent("""\
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import matplotlib.pyplot as plt""")

    enc_cols_repr = repr(encoded_cols)
    feat_names_repr = repr(feature_names)

    eval_cell_clf = textwrap.dedent(f"""\
        # ── Evaluate the model ──────────────────────────────────────────────────
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"Accuracy: {{acc*100:.2f}}%")
        print()
        print("Classification Report:")
        print(classification_report(y_test, preds))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greys')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()""")

    eval_cell_reg = textwrap.dedent(f"""\
        # ── Evaluate the model ──────────────────────────────────────────────────
        preds = model.predict(X_test)
        r2    = r2_score(y_test, preds)
        mse   = mean_squared_error(y_test, preds)
        mae   = mean_absolute_error(y_test, preds)
        print(f"R² Score : {{r2*100:.2f}}%")
        print(f"MSE      : {{mse:.4f}}")
        print(f"MAE      : {{mae:.4f}}")
        print(f"RMSE     : {{mse**0.5:.4f}}")

        # Actual vs Predicted
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds, alpha=0.5, color='#18181b', s=20)
        mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
        plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect fit')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()""")

    pred_input_example = '{' + ', '.join(
        [f'"{f}": 0' for f in feature_names[:3]] +
        (['...' ] if len(feature_names) > 3 else [])
    ) + '}'

    cells = [
        # ── Cell 0: Title ───────────────────────────────────────────────────────
        _md_cell(f"""\
# AutoAnalyst AI — Machine Learning Notebook
**Dataset:** `{dataset_name}`  
**Target Column:** `{target}`  
**Problem Type:** {problem_type.title()}  
**Best Model:** {best_model_name}  
**{metric_label}:** {best_score}%

---
This notebook was auto-generated by **AutoAnalyst AI**.  
It reproduces the complete ML pipeline: data loading → preprocessing → training → evaluation → prediction.
"""),
        # ── Cell 1: Imports ──────────────────────────────────────────────────────
        _code_cell(f"""\
# ── Imports ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
{imports_clf if problem_type == 'classification' else imports_reg}

print("Libraries loaded successfully.")
"""),
        # ── Cell 2: Load data ────────────────────────────────────────────────────
        _md_cell("## 1. Load & Inspect Dataset"),
        _code_cell(f"""\
# ── Load Dataset ─────────────────────────────────────────────────────────────────
# Update the path to point to your CSV/XLSX file
df = pd.read_csv("{dataset_name}")   # or pd.read_excel(...)

print(f"Shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print(f"Columns: {{list(df.columns)}}")
df.head(10)
"""),
        # ── Cell 3: EDA ─────────────────────────────────────────────────────────
        _md_cell("## 2. Exploratory Data Analysis"),
        _code_cell(f"""\
# ── Basic Statistics ──────────────────────────────────────────────────────────────
print("=== Shape ===")
print(f"Rows: {{df.shape[0]}}, Columns: {{df.shape[1]}}")

print("\\n=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({{'Missing Count': missing, 'Missing %': missing_pct}})
print(missing_df[missing_df['Missing Count'] > 0].to_string())
if missing_df['Missing Count'].sum() == 0:
    print("No missing values found!")

print(f"\\n=== Duplicates ===")
print(f"Duplicate rows: {{df.duplicated().sum()}}")

print("\\n=== Data Types ===")
print(df.dtypes.to_string())

print("\\n=== Statistical Summary ===")
df.describe(include='all')
"""),
        # ── Cell 4: Preprocessing ───────────────────────────────────────────────
        _md_cell("## 3. Data Preprocessing"),
        _code_cell(f"""\
# ── Step 1: Drop rows where target is null ────────────────────────────────────────
TARGET = "{target}"
df = df.dropna(subset=[TARGET])
print(f"Rows after dropping null target: {{len(df)}}")

# ── Step 2: Drop uninformative columns ───────────────────────────────────────────
SKIP_COLS = ['id', 'index', 'row', 'no', 'num', '#', 'uuid', 'email', 'phone', 'name']
cols_to_drop = []
for col in df.columns:
    if col == TARGET:
        continue
    if df[col].isna().mean() > 0.80:
        cols_to_drop.append(col)
    elif df[col].nunique() <= 1:
        cols_to_drop.append(col)
    elif col.lower() in SKIP_COLS:
        cols_to_drop.append(col)

if cols_to_drop:
    print(f"Dropping columns: {{cols_to_drop}}")
    df = df.drop(columns=cols_to_drop)

# ── Step 3: Impute missing values ────────────────────────────────────────────────
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [Numeric]  '{{col}}' → filled with median = {{median_val:.4f}}")
        else:
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"  [Categorical] '{{col}}' → filled with mode = '{{mode_val}}'")

# ── Step 4: Label encode categorical columns ──────────────────────────────────────
encoders = {{}}
label_mappings = {{}}
ENCODED_COLS = {enc_cols_repr}

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        label_mappings[col] = list(le.classes_)
        print(f"  Encoded '{{col}}': {{list(le.classes_)[:5]}}{'...' if len(le.classes_) > 5 else ''}")

# ── Step 5: Final coerce & split ─────────────────────────────────────────────────
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

FEATURE_NAMES = {feat_names_repr}
X = df[FEATURE_NAMES].astype(float)
y = df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\\nTrain size: {{len(X_train)}}, Test size: {{len(X_test)}}")
print(f"Features used: {{FEATURE_NAMES}}")
"""),
        # ── Cell 5: Train ───────────────────────────────────────────────────────
        _md_cell(f"## 4. Model Training — {best_model_name}"),
        _code_cell(f"""\
# ── Train the Best Model ──────────────────────────────────────────────────────────
model = {model_init}
model.fit(X_train, y_train)
print(f"Model trained: {{type(model).__name__}}")
print(f"Training samples: {{len(X_train)}}")
"""),
        # ── Cell 6: Evaluate ─────────────────────────────────────────────────────
        _md_cell("## 5. Model Evaluation"),
        _code_cell(eval_cell_clf if problem_type == 'classification' else eval_cell_reg),
        # ── Cell 7: Feature importance ───────────────────────────────────────────
        _md_cell("## 6. Feature Importance"),
        _code_cell(f"""\
# ── Feature Importance ────────────────────────────────────────────────────────────
feature_names = {feat_names_repr}

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_).flatten()
    if len(importances) != len(feature_names):
        importances = importances[:len(feature_names)]
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({{'Feature': feature_names, 'Importance': importances}})
    fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    fi_df['Rank'] = range(1, len(fi_df) + 1)
    print(fi_df.to_string(index=False))

    plt.figure(figsize=(8, max(4, len(feature_names) * 0.4)))
    plt.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='#18181b')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    print("Feature importance not available for this model type.")
"""),
        # ── Cell 8: Predict new sample ───────────────────────────────────────────
        _md_cell("## 7. Predict on New Data"),
        _code_cell(f"""\
# ── Predict on a new sample ───────────────────────────────────────────────────────
# Fill in the values for each feature below:
new_sample = {{
{chr(10).join(f'    "{f}": 0,  # {"categorical — choose from: " + str(label_mappings[f][:4]) if f in label_mappings else "numeric"}' for f in feature_names)}
}}

sample_df  = pd.DataFrame([new_sample])
sample_arr = scaler.transform(sample_df[feature_names].astype(float))
prediction = model.predict(sample_arr)[0]

# Decode label if categorical target
label_mappings_local = {repr(label_mappings)}
target_lm = label_mappings_local.get("{target}")
if target_lm is not None:
    try:
        prediction_label = target_lm[int(prediction)]
    except Exception:
        prediction_label = str(prediction)
else:
    prediction_label = str(prediction)

print(f"Predicted {target}: {{prediction_label}}")
"""),
        # ── Cell 9: Save model ───────────────────────────────────────────────────
        _md_cell("## 8. Save & Load Model"),
        _code_cell(f"""\
# ── Save the trained model ────────────────────────────────────────────────────────
import joblib

joblib.dump({{
    'model':          model,
    'scaler':         scaler,
    'encoders':       encoders,
    'label_mappings': label_mappings,
    'feature_names':  feature_names,
    'problem_type':   '{problem_type}',
    'target':         '{target}',
}}, 'best_model.joblib')

print("Model saved to 'best_model.joblib'")

# ── To reload and use later ──────────────────────────────────────────────────────
# loaded = joblib.load('best_model.joblib')
# model  = loaded['model']
# scaler = loaded['scaler']
"""),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "cells": cells
    }

    nb_json = json.dumps(notebook, indent=2, ensure_ascii=False)
    response = HttpResponse(nb_json, content_type='application/x-ipynb+json')
    base_name = dataset_name.rsplit('.', 1)[0]
    response['Content-Disposition'] = f'attachment; filename="autoanalyst_{base_name}_pipeline.ipynb"'
    return response


# ── Notebook helpers ───────────────────────────────────────────────────────────

def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip()
    }


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip()
    }
