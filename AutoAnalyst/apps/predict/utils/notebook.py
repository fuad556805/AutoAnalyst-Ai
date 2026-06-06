import json


def _md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md-{abs(hash(source[:40])) % 10**8}",
        "metadata": {},
        "source": source,
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"code-{abs(hash(source[:40])) % 10**8}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def generate_notebook(meta: dict, dataset_name: str) -> str:
    problem_type    = meta.get('problem_type', 'classification')
    target          = meta.get('target', 'target')
    best_model_name = meta.get('best_model_name', 'Random Forest')
    metric_label    = meta.get('metric_label', 'Accuracy')
    best_score      = meta.get('best_score', 0)
    feature_names   = meta.get('feature_names', [])
    label_mappings  = meta.get('label_mappings', {})
    pre             = meta.get('preprocessing', {})

    is_clf = problem_type == 'classification'

    # ── Column helpers ────────────────────────────────────────────────────────
    cat_cols = [f for f in feature_names if f in label_mappings]
    num_cols = [f for f in feature_names if f not in label_mappings]
    first_cat   = cat_cols[0]  if cat_cols else None
    second_cat  = cat_cols[1]  if len(cat_cols) > 1 else None
    third_cat   = cat_cols[2]  if len(cat_cols) > 2 else None
    first_num   = num_cols[0]  if num_cols else None
    second_num  = num_cols[1]  if len(num_cols) > 1 else None

    # ── Dataset reader ────────────────────────────────────────────────────────
    ext = dataset_name.rsplit('.', 1)[-1].lower() if '.' in dataset_name else 'csv'
    reader = (f'pd.read_csv("{dataset_name}")'
              if ext == 'csv' else f'pd.read_excel("{dataset_name}")')

    # ── All models for each problem type ──────────────────────────────────────
    if is_clf:
        all_models = [
            ('Logistic Regression',
             'LogisticRegression',
             'from sklearn.linear_model import LogisticRegression',
             'LogisticRegression(max_iter=2000, random_state=42)',
             'pred_lr'),
            ('Decision Tree',
             'DecisionTreeClassifier',
             'from sklearn.tree import DecisionTreeClassifier',
             'DecisionTreeClassifier(random_state=42)',
             'pred_dt'),
            ('Random Forest',
             'RandomForestClassifier',
             'from sklearn.ensemble import RandomForestClassifier',
             'RandomForestClassifier(n_estimators=100, random_state=42)',
             'pred_rf'),
            ('KNN',
             'KNeighborsClassifier',
             'from sklearn.neighbors import KNeighborsClassifier',
             'KNeighborsClassifier(n_neighbors=5)',
             'pred_knn'),
            ('Naive Bayes',
             'GaussianNB',
             'from sklearn.naive_bayes import GaussianNB',
             'GaussianNB()',
             'pred_nb'),
        ]
        model_var_map = {
            'Logistic Regression': 'pred_lr',
            'Decision Tree':       'pred_dt',
            'Random Forest':       'pred_rf',
            'K-Nearest Neighbors': 'pred_knn',
            'KNN':                 'pred_knn',
            'Support Vector Machine': 'pred_svm',
            'Naive Bayes':         'pred_nb',
        }
        best_pred_var = model_var_map.get(best_model_name, 'pred_rf')

        # Best model init (for Performance Metrics section)
        best_init_map = {
            'Logistic Regression':    'LogisticRegression(max_iter=2000, random_state=42)',
            'Decision Tree':          'DecisionTreeClassifier(random_state=42)',
            'Random Forest':          'RandomForestClassifier(n_estimators=100, random_state=42)',
            'K-Nearest Neighbors':    'KNeighborsClassifier(n_neighbors=5)',
            'KNN':                    'KNeighborsClassifier(n_neighbors=5)',
            'Support Vector Machine': 'SVC(random_state=42)',
            'Naive Bayes':            'GaussianNB()',
        }
        best_model_init = best_init_map.get(best_model_name, 'RandomForestClassifier(n_estimators=100, random_state=42)')

        # Reverse-find pred_var for best model
        for name, cls, imp, init, pvar in all_models:
            if name == best_model_name or cls in best_model_name:
                best_pred_var = pvar
                best_model_init = init
                break
    else:
        all_models = [
            ('Linear Regression',
             'LinearRegression',
             'from sklearn.linear_model import LinearRegression',
             'LinearRegression()',
             'pred_lr'),
            ('Ridge Regression',
             'Ridge',
             'from sklearn.linear_model import Ridge',
             'Ridge(alpha=1.0)',
             'pred_ridge'),
            ('Random Forest Regressor',
             'RandomForestRegressor',
             'from sklearn.ensemble import RandomForestRegressor',
             'RandomForestRegressor(n_estimators=100, random_state=42)',
             'pred_rf'),
        ]
        best_pred_var = 'pred_rf'
        best_model_init = 'RandomForestRegressor(n_estimators=100, random_state=42)'
        for name, cls, imp, init, pvar in all_models:
            if name == best_model_name:
                best_pred_var = pvar
                best_model_init = init
                break

    # ── All model names + pred vars (for Final View) ──────────────────────────
    model_names_list = [m[0] for m in all_models]
    pred_vars_list   = [m[4] for m in all_models]

    cells = []

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Final Project for Supervised Machine Learning'))
    cells.append(_md(f'# {target.replace("_", " ").title()} Prediction — {dataset_name}'))

    # ══════════════════════════════════════════════════════════════════════════
    # 1. IMPORTS
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_code(f'df = {reader}\ndf'))
    cells.append(_code("df.shape"))
    cells.append(_code("df.isnull().sum()"))
    cells.append(_code("df.duplicated().sum()"))

    # ── Target distribution & percentages (classification) ─────────────────
    if is_clf:
        cells.append(_code(f"sns.countplot(df['{target}'])\nplt.show()"))
        cells.append(_code(f"df['{target}'].value_counts()"))

        # Percentage per class
        target_classes = list(label_mappings.get(target, []))
        if len(target_classes) >= 2:
            c1, c2 = target_classes[0], target_classes[1]
            cells.append(_code(
                f"class_1 = df[df['{target}'] == '{c1}'].shape[0]\n"
                f"class_2 = df[df['{target}'] == '{c2}'].shape[0]\n"
                f"\n"
                f"pct_1 = (class_1 / (class_1 + class_2)) * 100\n"
                f"pct_2 = (class_2 / (class_1 + class_2)) * 100\n"
                f"\n"
                f"print(f'Percentage — {c1}: {{pct_1:.2f}}%')\n"
                f"print(f'Percentage — {c2}: {{pct_2:.2f}}%')"
            ))
        else:
            cells.append(_code(
                f"for cls in df['{target}'].unique():\n"
                f"    pct = (df[df['{target}'] == cls].shape[0] / len(df)) * 100\n"
                f"    print(f'{{cls}}: {{pct:.2f}}%')"
            ))

    cells.append(_code("df.describe()"))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Visualization'))
    cells.append(_code("df.head(4)"))

    if is_clf:
        if first_cat:
            cells.append(_code(
                f"sns.countplot(x='{first_cat}', hue='{target}', data=df)\n"
                f"plt.show()"
            ))
            cells.append(_code(f"df['{first_cat}'].value_counts()"))

        if first_num:
            cells.append(_code(
                f"sns.histplot(data=df, x='{first_num}', hue='{target}', bins=30)\n"
                f"plt.show()"
            ))

        if first_num:
            cells.append(_code(
                f"sns.boxplot(x='{first_num}', hue='{target}', data=df)\n"
                f"plt.show()"
            ))

        if second_cat:
            cells.append(_code(
                f"sns.countplot(x='{second_cat}', hue='{target}', data=df)\n"
                f"plt.show()"
            ))

        if third_cat:
            cells.append(_code(
                f"sns.countplot(x='{third_cat}', hue='{target}', data=df)\n"
                f"plt.show()"
            ))

        if second_num:
            cells.append(_code(
                f"sns.histplot(data=df, x='{second_num}', hue='{target}', bins=30)\n"
                f"plt.show()"
            ))
    else:
        if first_num:
            cells.append(_code(
                f"sns.histplot(data=df, x='{first_num}', bins=30)\n"
                f"plt.show()"
            ))
        if first_cat:
            cells.append(_code(
                f"sns.countplot(x='{first_cat}', data=df)\n"
                f"plt.show()"
            ))
        cells.append(_code(
            f"sns.histplot(data=df, x='{target}', bins=30)\n"
            f"plt.show()"
        ))
        if first_num and second_num:
            cells.append(_code(
                f"sns.scatterplot(x='{first_num}', y='{target}', data=df)\n"
                f"plt.show()"
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. DATA PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Data Preprocessing'))

    keep_cols = feature_names + [target]
    drop_cols = [c for c in ['CustomerID', 'ID', 'id', 'Index']
                 if c not in keep_cols]
    if drop_cols:
        cells.append(_code(
            f"# Drop ID-like and uninformative columns\n"
            f"data = df.drop({repr(drop_cols)}, axis=1, errors='ignore')\n"
            f"data.head(5)"
        ))
    else:
        cells.append(_code(
            f"# Keep only the relevant features and target\n"
            f"data = df[{repr(keep_cols)}].copy()\n"
            f"data.head(5)"
        ))

    cells.append(_code("data.shape"))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. LABEL ENCODING
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Label Encoding'))
    cells.append(_code(
        "from sklearn.preprocessing import LabelEncoder\n"
        "\n"
        "for column in data.columns:\n"
        "    if data[column].dtype == 'object':\n"
        "        data[column] = LabelEncoder().fit_transform(data[column])\n"
        "\n"
        "data"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. TRAIN AND TEST DATA
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Train and Test Data'))
    cells.append(_code(
        f'x = data.drop(\'{target}\', axis=1)\n'
        f'y = data[\'{target}\']\n'
        f'\n'
        f'y'
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 7. SCALING
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Scalling the data'))
    cells.append(_code(
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "feature_x = StandardScaler().fit_transform(x)\n"
        "\n"
        "feature_x"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SPLIT DATA
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Split Data'))
    cells.append(_code(
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "xtrain, xtest, ytrain, ytest = train_test_split(\n"
        "    feature_x, y, test_size=0.25, random_state=42\n"
        ")\n"
        "\n"
        "xtest"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 9. EACH MODEL (classification)
    # ══════════════════════════════════════════════════════════════════════════
    if is_clf:
        cells.append(_md('# Performance Imports'))
        cells.append(_code(
            "from sklearn.metrics import accuracy_score\n"
            "from sklearn.metrics import confusion_matrix\n"
            "from sklearn.metrics import classification_report"
        ))

        for model_name, cls_name, import_line, model_init, pred_var in all_models:
            cells.append(_md(f'# {model_name} Model'))
            cells.append(_code(
                f"{import_line}\n"
                f"\n"
                f"model_{pred_var} = {model_init}\n"
                f"model_{pred_var}.fit(xtrain, ytrain)\n"
                f"\n"
                f"{pred_var} = model_{pred_var}.predict(xtest)\n"
                f"{pred_var}.shape"
            ))
            cells.append(_code(f"accuracy_score(ytest, {pred_var})"))

        # ── Performance Metrics (best model) ─────────────────────────────────
        cells.append(_md('# Performance Metrics'))
        cells.append(_code(
            f"# Accuracy score — {best_model_name}\n"
            f"accuracy_score(ytest, {best_pred_var})"
        ))
        cells.append(_code(
            f"# Confusion Matrix — {best_model_name}\n"
            f"confusion_matrix(ytest, {best_pred_var})"
        ))
        cells.append(_code(
            f"# Classification Report — {best_model_name}\n"
            f"print(classification_report(ytest, {best_pred_var}))"
        ))

    else:
        # ── Regression models ─────────────────────────────────────────────────
        cells.append(_md('# Performance Imports'))
        cells.append(_code(
            "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error"
        ))

        for model_name, cls_name, import_line, model_init, pred_var in all_models:
            cells.append(_md(f'# {model_name} Model'))
            cells.append(_code(
                f"{import_line}\n"
                f"\n"
                f"model_{pred_var} = {model_init}\n"
                f"model_{pred_var}.fit(xtrain, ytrain)\n"
                f"\n"
                f"{pred_var} = model_{pred_var}.predict(xtest)"
            ))
            cells.append(_code(
                f"r2 = r2_score(ytest, {pred_var})\n"
                f"print(f'R²: {{r2*100:.2f}}%')"
            ))

        cells.append(_md('# Performance Metrics'))
        cells.append(_code(
            f"# Detailed metrics — {best_model_name}\n"
            f"r2   = r2_score(ytest, {best_pred_var})\n"
            f"mse  = mean_squared_error(ytest, {best_pred_var})\n"
            f"mae  = mean_absolute_error(ytest, {best_pred_var})\n"
            f"rmse = mse ** 0.5\n"
            f"\n"
            f"print(f'R²  : {{r2*100:.2f}}%')\n"
            f"print(f'MAE : {{mae:.4f}}')\n"
            f"print(f'MSE : {{mse:.4f}}')\n"
            f"print(f'RMSE: {{rmse:.4f}}')"
        ))
        cells.append(_code(
            f"plt.figure(figsize=(7, 5))\n"
            f"plt.scatter(ytest, {best_pred_var}, alpha=0.5, color='steelblue', s=20)\n"
            f"mn = min(ytest.min(), {best_pred_var}.min())\n"
            f"mx = max(ytest.max(), {best_pred_var}.max())\n"
            f"plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect fit')\n"
            f"plt.xlabel('Actual {target}')\n"
            f"plt.ylabel('Predicted {target}')\n"
            f"plt.title('Actual vs Predicted — {best_model_name}')\n"
            f"plt.legend()\n"
            f"plt.tight_layout()\n"
            f"plt.show()"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 10. FINAL VIEW — all models comparison
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md('# Final View'))

    if is_clf:
        score_entries = ',\n        '.join(
            f'accuracy_score(ytest, {pv})' for pv in pred_vars_list
        )
        names_repr = repr(model_names_list)
        cells.append(_code(
            f"models = pd.DataFrame({{\n"
            f"    'Model': {names_repr},\n"
            f"    'Accuracy': [\n"
            f"        {score_entries}\n"
            f"    ]\n"
            f"}})\n"
            f"\n"
            f"models"
        ))
    else:
        score_entries = ',\n        '.join(
            f'r2_score(ytest, {pv}) * 100' for pv in pred_vars_list
        )
        names_repr = repr(model_names_list)
        cells.append(_code(
            f"models = pd.DataFrame({{\n"
            f"    'Model': {names_repr},\n"
            f"    'R2_Score (%)': [\n"
            f"        {score_entries}\n"
            f"    ]\n"
            f"}})\n"
            f"\n"
            f"models"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 11. ACCURACY GRAPH
    # ══════════════════════════════════════════════════════════════════════════
    y_col = 'Accuracy' if is_clf else 'R2_Score (%)'
    cells.append(_md('# Accuracy Graph:'))
    cells.append(_code(
        f"sns.barplot(x='Model', y='{y_col}', data=models)\n"
        f"plt.xticks(rotation=45)\n"
        f"plt.title('Model Comparison')\n"
        f"plt.tight_layout()\n"
        f"plt.show()"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 12. CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    if is_clf:
        conclusion_text = (
            f'# Conclusion:\n\n'
            f'In this dataset, **{best_model_name}** achieved the highest accuracy '
            f'(**{best_score}%**).\n\n'
        )
        for name in model_names_list:
            if name != best_model_name:
                conclusion_text += f'- **{name}** was also evaluated and its results are shown in the comparison above.\n'
        conclusion_text += (
            f'\nFeature importance analysis from the best model shows which features '
            f'have the most impact on predicting **{target}**.\n\n'
            f'With hyperparameter tuning, **{best_model_name}** could potentially '
            f'achieve even better performance in the future.'
        )
    else:
        conclusion_text = (
            f'# Conclusion:\n\n'
            f'In this regression task, **{best_model_name}** achieved the best R² score '
            f'(**{best_score}%**).\n\n'
        )
        for name in model_names_list:
            if name != best_model_name:
                conclusion_text += f'- **{name}** was also evaluated and its results are shown in the comparison above.\n'
        conclusion_text += (
            f'\nWith further hyperparameter tuning, **{best_model_name}** could '
            f'potentially achieve even better performance.'
        )

    cells.append(_md(conclusion_text))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }
    return json.dumps(notebook, indent=2, ensure_ascii=False)
