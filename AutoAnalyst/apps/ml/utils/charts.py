import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


CHART_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#d4d4d8',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#09090b',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#e4e4e7',
    'grid.linestyle': '--',
}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def chart_confusion(y_test, preds, labels=None) -> str:
    cm = confusion_matrix(y_test, preds, labels=labels)
    n = len(cm)
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(max(4.5, n * 1.2), max(3.8, n * 1.1)))
        im = ax.imshow(cm, cmap='Greys', vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        if labels is not None:
            ax.set_xticklabels([str(l) for l in labels], rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels([str(l) for l in labels], fontsize=9)
        thresh = cm.max() * 0.5
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                ax.text(j, i, f'{val:,}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if val > thresh else '#09090b')
        ax.set_xlabel('Predicted Label', fontsize=9, labelpad=8)
        ax.set_ylabel('True Label', fontsize=9, labelpad=8)
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_actual_vs_pred(y_test, preds) -> str:
    y_arr = np.array(y_test)
    p_arr = np.array(preds)
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(y_arr, p_arr, alpha=0.45, color='#18181b', s=18, zorder=3, label='Samples')
        mn = min(float(y_arr.min()), float(p_arr.min()))
        mx = max(float(y_arr.max()), float(p_arr.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.8, label='Perfect fit', zorder=4)
        # Residual shading
        ax.fill_between([mn, mx], [mn * 0.95, mx * 0.95], [mn * 1.05, mx * 1.05],
                        alpha=0.08, color='gray', label='±5% band')
        ax.set_xlabel('Actual Values', fontsize=9)
        ax.set_ylabel('Predicted Values', fontsize=9)
        ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold', pad=12)
        ax.legend(fontsize=8)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_feature_importance(feature_names, importances, top_n=12) -> str:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]
    norm  = [v / max(vals) for v in vals]

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, max(3.2, len(names) * 0.45)))
        colors = [plt.cm.Greys(0.3 + 0.65 * n) for n in norm]
        bars = ax.barh(names, vals, color=colors, edgecolor='none', height=0.65)
        ax.set_xlabel('Importance Score', fontsize=9)
        ax.set_title('Feature Importance (Top Features)', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8, fontweight='600')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_model_comparison(names, scores, metric_label, best_name) -> str:
    pairs = sorted(zip(names, scores), key=lambda x: x[1])
    s_names = [p[0] for p in pairs]
    s_scores = [p[1] for p in pairs]
    colors = ['#09090b' if n == best_name else '#a1a1aa' for n in s_names]

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, max(3.2, len(names) * 0.65)))
        bars = ax.barh(s_names, s_scores, color=colors, edgecolor='none', height=0.6)
        ax.set_xlabel(f'{metric_label} (%)', fontsize=9)
        ax.set_title('Model Comparison', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlim(0, max(s_scores) * 1.15 if s_scores else 100)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        for bar, val, name in zip(bars, s_scores, s_names):
            label = f'{val}%  ★ Best' if name == best_name else f'{val}%'
            ax.text(bar.get_width() + 0.4,
                    bar.get_y() + bar.get_height() / 2,
                    label, va='center', fontsize=8,
                    fontweight='bold' if name == best_name else 'normal')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)
