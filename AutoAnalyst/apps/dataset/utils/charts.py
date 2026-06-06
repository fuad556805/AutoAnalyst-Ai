import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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


def build_visualizations(df: pd.DataFrame, target: str) -> list:
    charts = []

    def add(title, chart_type, b64):
        charts.append({'title': title, 'type': chart_type, 'img': b64})

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ── Target distribution ────────────────────────────────────────────────────
    if target and target in df.columns:
        with plt.rc_context(CHART_STYLE):
            if pd.api.types.is_numeric_dtype(df[target]):
                fig, ax = plt.subplots(figsize=(6, 3.5))
                data_t = df[target].dropna()
                ax.hist(data_t, bins=min(30, max(10, len(data_t) // 20)),
                        color='#18181b', edgecolor='white', linewidth=0.6, alpha=0.85)
                ax.set_xlabel(target, fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
            else:
                vc = df[target].value_counts()
                if len(vc) <= 8:
                    # Pie chart for small number of categories
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    wedges, texts, autotexts = ax.pie(
                        vc.values,
                        labels=vc.index.astype(str),
                        autopct='%1.1f%%',
                        colors=[plt.cm.Greys(i / (len(vc) + 1) + 0.2) for i in range(len(vc))],
                        startangle=140,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                    for at in autotexts:
                        at.set_fontsize(9)
                        at.set_fontweight('bold')
                    ax.set_ylabel('')
                else:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    bars = ax.barh(vc.index.astype(str)[:12], vc.values[:12],
                                   color='#18181b', edgecolor='none', height=0.6, alpha=0.88)
                    ax.set_xlabel('Count', fontsize=9)
                    for bar, val in zip(bars, vc.values[:12]):
                        ax.text(bar.get_width() + vc.values.max() * 0.01,
                                bar.get_y() + bar.get_height() / 2,
                                f'{val:,}', va='center', fontsize=8)
            ax.set_title(f'Target Distribution — {target}', fontsize=11, fontweight='bold', pad=10)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(f'Target — {target}', 'target', fig_to_b64(fig))

    # ── Box plots: numeric vs target (if target is categorical) ───────────────
    if target and target in df.columns and not pd.api.types.is_numeric_dtype(df[target]):
        target_vals = df[target].dropna().unique()
        if len(target_vals) <= 10:
            for col in num_cols[:5]:
                if col == target:
                    continue
                with plt.rc_context(CHART_STYLE):
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    groups = [df[df[target] == v][col].dropna().values for v in target_vals]
                    bp = ax.boxplot(groups, labels=[str(v) for v in target_vals],
                                   patch_artist=True, notch=False,
                                   medianprops={'color': 'black', 'linewidth': 2},
                                   whiskerprops={'color': '#52525b'},
                                   capprops={'color': '#52525b'},
                                   flierprops={'marker': 'o', 'markersize': 3,
                                               'markerfacecolor': '#a1a1aa', 'alpha': 0.5})
                    grays = [plt.cm.Greys(0.2 + 0.6 * i / max(len(groups) - 1, 1))
                             for i in range(len(groups))]
                    for patch, color in zip(bp['boxes'], grays):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.75)
                    ax.set_xlabel(target, fontsize=9)
                    ax.set_ylabel(col, fontsize=9)
                    ax.set_title(f'{col} by {target}', fontsize=11, fontweight='bold')
                    ax.tick_params(axis='x', rotation=30 if len(target_vals) > 4 else 0)
                    fig.patch.set_facecolor('white')
                    plt.tight_layout()
                add(f'{col} vs {target}', 'boxplot', fig_to_b64(fig))

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        heat_cols = [c for c in num_cols if c != target][:12]
        corr = df[heat_cols].corr()
        n = len(heat_cols)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(max(5.5, n * 0.8), max(4.5, n * 0.75)))
            im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=7)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(heat_cols, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(heat_cols, fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = corr.values[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white' if abs(val) > 0.55 else '#18181b')
            ax.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold', pad=10)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add('Correlation Matrix', 'heatmap', fig_to_b64(fig))

    # ── Numeric histograms with KDE ───────────────────────────────────────────
    for col in num_cols[:8]:
        if col == target:
            continue
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            data = df[col].dropna()
            n_bins = min(30, max(10, len(data) // 20))
            ax.hist(data, bins=n_bins, color='#18181b', edgecolor='white',
                    linewidth=0.5, alpha=0.75, density=True)
            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data, bw_method='scott')
                xs = np.linspace(data.min(), data.max(), 200)
                ax.plot(xs, kde(xs), color='#18181b', linewidth=2, label='KDE')
            except Exception:
                pass
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(col, 'histogram', fig_to_b64(fig))

    # ── Categorical bar charts (horizontal for readability) ───────────────────
    for col in cat_cols[:6]:
        if col == target:
            continue
        vc = df[col].value_counts().head(12)
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(5, max(2.8, len(vc) * 0.35)))
            bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                           color='#18181b', edgecolor='none', height=0.65, alpha=0.85)
            ax.set_xlabel('Count', fontsize=9)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            for bar, val in zip(bars, vc.values[::-1]):
                ax.text(bar.get_width() + vc.values.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:,}', va='center', fontsize=8)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
        add(col, 'bar', fig_to_b64(fig))

    # ── Pie for small-cardinality categoricals ────────────────────────────────
    for col in cat_cols[:4]:
        if col == target:
            continue
        vc = df[col].value_counts()
        if 2 <= len(vc) <= 6:
            with plt.rc_context(CHART_STYLE):
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                wedges, texts, autotexts = ax.pie(
                    vc.values,
                    labels=vc.index.astype(str),
                    autopct='%1.1f%%',
                    colors=[plt.cm.Greys(0.15 + 0.7 * i / max(len(vc) - 1, 1))
                            for i in range(len(vc))],
                    startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                )
                for at in autotexts:
                    at.set_fontsize(8)
                    at.set_fontweight('bold')
                ax.set_title(f'{col} — Proportions', fontsize=11, fontweight='bold')
                ax.set_ylabel('')
                fig.patch.set_facecolor('white')
                plt.tight_layout()
            add(f'{col} (pie)', 'pie', fig_to_b64(fig))

    return charts
