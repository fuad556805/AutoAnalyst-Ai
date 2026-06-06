import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def build_visualizations(df: pd.DataFrame, target: str) -> list:
    charts = []

    def add(title, chart_type, b64):
        charts.append({'title': title, 'type': chart_type, 'img': b64})

    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols[:8]:
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            data = df[col].dropna()
            ax.hist(data, bins=min(30, max(10, len(data) // 20)),
                    color='#18181b', edgecolor='white', linewidth=0.5, alpha=0.88)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(col, 'histogram', fig_to_b64(fig))

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(12)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.bar(vc.index.astype(str), vc.values,
                   color='#18181b', edgecolor='none', width=0.6, alpha=0.88)
            ax.set_title(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=40)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(col, 'bar', fig_to_b64(fig))

    if len(num_cols) >= 2:
        heat_cols = num_cols[:12]
        corr = df[heat_cols].corr()
        n = len(heat_cols)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(max(5, n * 0.75), max(4, n * 0.7)))
            im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(heat_cols, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(heat_cols, fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = corr.values[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(val) > 0.6 else '#09090b')
            ax.set_title('Feature Correlation Matrix')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add('Correlation Matrix', 'heatmap', fig_to_b64(fig))

    if target and target in df.columns:
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            if pd.api.types.is_numeric_dtype(df[target]):
                ax.hist(df[target].dropna(), bins=20, color='#16a34a',
                        edgecolor='white', linewidth=0.5, alpha=0.88)
                ax.set_xlabel(target)
                ax.set_ylabel('Count')
            else:
                vc = df[target].value_counts()
                ax.bar(vc.index.astype(str), vc.values, color='#16a34a',
                       edgecolor='none', width=0.6, alpha=0.88)
                ax.tick_params(axis='x', rotation=40)
            ax.set_title(f'Target: {target}')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(f'Target — {target}', 'target', fig_to_b64(fig))

    return charts
