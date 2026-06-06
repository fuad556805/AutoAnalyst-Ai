import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                              mean_absolute_error, precision_score,
                              recall_score, f1_score)

from .charts import (chart_confusion, chart_actual_vs_pred,
                     chart_feature_importance, chart_model_comparison)


def _build_feature_importance(model, feature_names: list) -> tuple:
    fi_list = []
    fi_chart = None
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(np.array(model.coef_).flatten())
        else:
            return fi_list, fi_chart

        pairs = sorted(zip(feature_names, importances),
                       key=lambda x: x[1], reverse=True)[:10]
        fi_list = [{'feature': f, 'importance': round(float(v), 4)} for f, v in pairs]
        fi_chart = chart_feature_importance(
            [p[0] for p in pairs], [p[1] for p in pairs])
    except Exception:
        pass
    return fi_list, fi_chart


def run_classification(X_train, X_test, y_train, y_test, feature_names: list) -> dict:
    metric_label = 'Accuracy'
    model_defs = [
        ('Logistic Regression',    LogisticRegression(max_iter=2000, random_state=42)),
        ('Random Forest',          RandomForestClassifier(n_estimators=100, random_state=42)),
        ('K-Nearest Neighbors',    KNeighborsClassifier(n_neighbors=min(5, len(X_train)))),
        ('Support Vector Machine', SVC(random_state=42)),
    ]

    results = []
    for name, model in model_defs:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = round(accuracy_score(y_test, preds) * 100, 2)
            results.append({'name': name, 'score': score, 'model': model, 'preds': preds})
        except Exception:
            results.append({'name': name, 'score': 0.0, 'model': None, 'preds': None})

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    charts_data = {}
    if best.get('preds') is not None:
        try:
            labels = sorted(y_test.unique())
            charts_data['confusion'] = chart_confusion(y_test, best['preds'], labels=labels)
        except Exception:
            pass

    extra_metrics = {}
    if best.get('preds') is not None:
        try:
            avg = 'binary' if len(y_test.unique()) == 2 else 'weighted'
            extra_metrics = {
                'Precision': round(precision_score(y_test, best['preds'], average=avg, zero_division=0) * 100, 2),
                'Recall':    round(recall_score(y_test,    best['preds'], average=avg, zero_division=0) * 100, 2),
                'F1 Score':  round(f1_score(y_test,        best['preds'], average=avg, zero_division=0) * 100, 2),
            }
        except Exception:
            pass

    fi_list, fi_chart = _build_feature_importance(best['model'], feature_names)
    if fi_chart:
        charts_data['feature_importance'] = fi_chart

    try:
        names = [r['name'] for r in results]
        scores = [r['score'] for r in results]
        charts_data['model_comparison'] = chart_model_comparison(
            names, scores, metric_label, best['name'])
    except Exception:
        pass

    return {
        'results': results,
        'best': best,
        'charts_data': charts_data,
        'extra_metrics': extra_metrics,
        'feature_importance': fi_list,
        'metric_label': metric_label,
    }


def run_regression(X_train, X_test, y_train, y_test, feature_names: list) -> dict:
    metric_label = 'R² Score'
    model_defs = [
        ('Linear Regression',       LinearRegression()),
        ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Ridge Regression',        Ridge()),
    ]

    results = []
    for name, model in model_defs:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = round(max(r2_score(y_test, preds), 0) * 100, 2)
            results.append({'name': name, 'score': score, 'model': model, 'preds': preds})
        except Exception:
            results.append({'name': name, 'score': 0.0, 'model': None, 'preds': None})

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    charts_data = {}
    if best.get('preds') is not None:
        try:
            charts_data['actual_vs_pred'] = chart_actual_vs_pred(y_test, best['preds'])
        except Exception:
            pass

    extra_metrics = {}
    if best.get('preds') is not None:
        try:
            extra_metrics = {
                'MAE':  round(float(mean_absolute_error(y_test, best['preds'])), 4),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test, best['preds']))), 4),
                'MSE':  round(float(mean_squared_error(y_test, best['preds'])), 4),
            }
        except Exception:
            pass

    fi_list, fi_chart = _build_feature_importance(best['model'], feature_names)
    if fi_chart:
        charts_data['feature_importance'] = fi_chart

    try:
        names = [r['name'] for r in results]
        scores = [r['score'] for r in results]
        charts_data['model_comparison'] = chart_model_comparison(
            names, scores, metric_label, best['name'])
    except Exception:
        pass

    return {
        'results': results,
        'best': best,
        'charts_data': charts_data,
        'extra_metrics': extra_metrics,
        'feature_importance': fi_list,
        'metric_label': metric_label,
    }


def generate_insights(best: dict, results: list, dropped_cols: list,
                      label_mappings: dict, problem_type: str,
                      extra_metrics: dict, n_train: int,
                      n_test: int, metric_label: str) -> list:
    insights = []
    try:
        insights.append(
            f"Best model: <strong>{best['name']}</strong> with "
            f"{metric_label} = <strong>{best['score']}%</strong>"
        )
        if len(results) > 1:
            worst = min(results, key=lambda x: x['score'])
            diff = round(best['score'] - worst['score'], 2)
            if diff > 0:
                insights.append(
                    f"Performance spread: <strong>{diff}%</strong> difference "
                    f"between best and worst model."
                )
        if dropped_cols:
            preview = ', '.join(dropped_cols[:4])
            suffix = '...' if len(dropped_cols) > 4 else ''
            insights.append(
                f"Auto-cleaned: removed <strong>{len(dropped_cols)}</strong> "
                f"column(s) (<em>{preview}{suffix}</em>) due to >80% missing or constant values."
            )
        if label_mappings:
            cols_preview = ', '.join(list(label_mappings.keys())[:5])
            insights.append(
                f"<strong>{len(label_mappings)}</strong> categorical column(s) "
                f"were label-encoded: {cols_preview}."
            )
        if problem_type == 'classification':
            if extra_metrics.get('F1 Score', 100) < 70:
                insights.append(
                    "⚠ F1 Score is below 70% — consider gathering more data "
                    "or trying feature engineering."
                )
        else:
            if best['score'] < 50:
                insights.append(
                    "⚠ R² Score is below 50% — the target may have low linear "
                    "predictability. Try adding more features."
                )
        insights.append(
            f"Train/test split: <strong>{n_train}</strong> training rows · "
            f"<strong>{n_test}</strong> test rows (80/20 split)."
        )
    except Exception:
        pass
    return insights
