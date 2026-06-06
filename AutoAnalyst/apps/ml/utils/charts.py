import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


CHART_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#e4e4e7',
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
    'grid.alpha': 0.3,
    'grid.color': '#e4e4e7',
}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def chart_confusion(y_test, preds, labels=None) -> str:
    cm = confusion_matrix(y_test, preds, labels=labels)
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.imshow(cm, cmap='Greys')
        ax.set_xticks(range(len(cm)))
        ax.set_yticks(range(len(cm)))
        if labels is not None:
            ax.set_xticklabels([str(l) for l in labels], rotation=45, ha='right')
            ax.set_yticklabels([str(l) for l in labels])
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() * 0.5 else '#09090b',
                        fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_actual_vs_pred(y_test, preds) -> str:
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, preds, alpha=0.5, color='#18181b', s=20, zorder=3)
        mn = min(float(y_test.min()), float(preds.min()))
        mx = max(float(y_test.max()), float(preds.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        ax.legend(fontsize=8)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_feature_importance(feature_names, importances, top_n=10) -> str:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.4)))
        bars = ax.barh(names, vals, color='#18181b', edgecolor='none', height=0.6)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        ax.set_xlim(0, max(vals) * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)


def chart_model_comparison(names, scores, metric_label, best_name) -> str:
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.55)))
        colors = ['#16a34a' if n == best_name else '#18181b' for n in names]
        bars = ax.barh(names, scores, color=colors, edgecolor='none', height=0.55)
        ax.set_xlabel(f'{metric_label} (%)')
        ax.set_title('Model Comparison')
        ax.set_xlim(0, max(scores) * 1.15 if scores else 100)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val}%', va='center', fontsize=8, fontweight='bold')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
    return fig_to_b64(fig)
