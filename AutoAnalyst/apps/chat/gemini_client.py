import os
import io
import re
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dataset(dataset_path):
    if not dataset_path or not os.path.exists(dataset_path):
        return None, 'No dataset loaded'
    try:
        ext = dataset_path.rsplit('.', 1)[-1].lower()
        df = pd.read_csv(dataset_path) if ext == 'csv' else pd.read_excel(dataset_path)
        return df, None
    except Exception as e:
        return None, str(e)


def build_dataset_context(df):
    if df is None:
        return None
    lines = [f'Shape: {df.shape[0]:,} rows × {df.shape[1]} columns', '', 'Columns:']
    for col in df.columns:
        dtype   = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        unique  = int(df[col].nunique())
        info    = f'  • {col} ({dtype}): {unique} unique, {missing} missing'
        if pd.api.types.is_numeric_dtype(df[col]):
            mn = df[col].min()
            mx = df[col].max()
            mu = df[col].mean()
            if not pd.isna(mn):
                info += f', range=[{mn:.4g}–{mx:.4g}], mean={mu:.4g}'
        else:
            tops = df[col].value_counts().head(3).index.tolist()
            info += f', top values: {tops}'
        lines.append(info)
    lines += ['', 'Sample (first 5 rows):']
    try:
        lines.append(df.head(5).fillna('').to_string(max_cols=12, max_colwidth=20))
    except Exception:
        pass
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        lines += ['', 'Numeric stats:']
        try:
            lines.append(df[numeric_cols[:8]].describe().round(4).to_string())
        except Exception:
            pass
    return '\n'.join(lines)


def build_system_prompt(dataset_context, dataset_name='dataset', ml_context=None):
    prompt = (
        'You are AutoAnalyst AI — an expert data science and ML assistant embedded in an AutoML platform.\n\n'
        'Style rules:\n'
        '- Be concise and precise. Get to the point.\n'
        '- Use **bold** for key numbers and terms.\n'
        '- Use bullet points and numbered lists for clarity.\n'
        '- For code/SQL, always use ```python or ```sql fenced blocks.\n'
        '- Never invent column names or statistics — only reference what you can see.\n'
        '- If you are unsure, say so.'
    )
    if dataset_context:
        prompt += f'\n\nThe user has uploaded "{dataset_name}":\n\n{dataset_context}'
    else:
        prompt += '\n\nThe user has not uploaded a dataset yet. Answer general ML/data science questions.'
    if ml_context:
        prompt += (
            f'\n\nTrained model info:\n'
            f'  • Best model: {ml_context.get("best_model_name", "?")}\n'
            f'  • Problem type: {ml_context.get("problem_type", "?")}\n'
            f'  • Best score: {ml_context.get("best_score", "?")} ({ml_context.get("metric_label", "")})'
        )
    prompt += (
        '\n\nSpecial output commands (place at the VERY END of your response, on a separate line):\n'
        '• To render a chart: [CHART:type:col1] or [CHART:type:col1:col2]\n'
        '  Valid types: bar, hist, scatter, line, pie, box, heatmap\n'
        '  Examples: [CHART:bar:species]  [CHART:scatter:age:salary]  [CHART:heatmap]\n'
        '• To offer a PDF report download: [PDF_REPORT]\n'
        'Only emit one command per response. Do not emit a command unless the user explicitly asks for a chart or report.'
    )
    return prompt


def parse_response(text):
    chart_spec     = None
    pdf_requested  = False

    chart_match = re.search(r'\[CHART:(\w+)(?::([^:\]\s]+))?(?::([^\]\s]+))?\]', text)
    if chart_match:
        chart_spec = {
            'type': chart_match.group(1).lower(),
            'col1': (chart_match.group(2) or '').strip(),
            'col2': (chart_match.group(3) or '').strip() or None,
        }
        text = text[:chart_match.start()].rstrip()

    if '[PDF_REPORT]' in text:
        pdf_requested = True
        text = text.replace('[PDF_REPORT]', '').rstrip()

    return text.strip(), chart_spec, pdf_requested


def generate_chart(df, chart_type, col1, col2=None):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fafb')
    for spine in ax.spines.values():
        spine.set_color('#e4e4e7')
    ax.grid(True, alpha=0.35, linewidth=0.5, color='#e4e4e7')
    ax.tick_params(colors='#71717a', labelsize=9)
    C = '#09090b'

    try:
        if chart_type == 'hist' and col1 and col1 in df.columns:
            data = df[col1].dropna()
            bins = min(30, max(10, len(data) // 20))
            ax.hist(data, bins=bins, color=C, alpha=0.82, edgecolor='white', linewidth=0.4)
            ax.set_title(f'Distribution — {col1}', fontsize=12, fontweight='bold', pad=10, color=C)
            ax.set_xlabel(col1, fontsize=9, color='#3f3f46')
            ax.set_ylabel('Frequency', fontsize=9, color='#3f3f46')

        elif chart_type == 'bar' and col1 and col1 in df.columns:
            vc = df[col1].value_counts().head(12)
            ax.bar(range(len(vc)), vc.values, color=C, alpha=0.82, width=0.72)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels([str(x)[:14] for x in vc.index], rotation=38, ha='right', fontsize=8)
            ax.set_title(f'{col1} — Value Counts', fontsize=12, fontweight='bold', pad=10, color=C)
            ax.set_ylabel('Count', fontsize=9, color='#3f3f46')

        elif chart_type == 'scatter' and col1 and col2 and col1 in df.columns and col2 in df.columns:
            ax.scatter(df[col1], df[col2], color=C, alpha=0.35, s=18, linewidths=0)
            ax.set_title(f'{col1} vs {col2}', fontsize=12, fontweight='bold', pad=10, color=C)
            ax.set_xlabel(col1, fontsize=9, color='#3f3f46')
            ax.set_ylabel(col2, fontsize=9, color='#3f3f46')

        elif chart_type == 'line' and col1 and col1 in df.columns:
            data = df[col1].dropna().reset_index(drop=True)
            ax.plot(data.values, color=C, linewidth=1.6, alpha=0.9)
            ax.set_title(f'{col1} — Trend', fontsize=12, fontweight='bold', pad=10, color=C)
            ax.set_xlabel('Index', fontsize=9, color='#3f3f46')
            ax.set_ylabel(col1, fontsize=9, color='#3f3f46')

        elif chart_type == 'pie' and col1 and col1 in df.columns:
            vc = df[col1].value_counts().head(7)
            palette = ['#09090b','#374151','#6b7280','#9ca3af','#d1d5db','#e5e7eb','#f3f4f6']
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=[str(x)[:14] for x in vc.index],
                autopct='%1.1f%%', colors=palette[:len(vc)],
                startangle=90, pctdistance=0.82,
                wedgeprops=dict(edgecolor='white', linewidth=1.2),
            )
            for t in autotexts:
                t.set_fontsize(8)
            ax.set_title(f'{col1} Distribution', fontsize=12, fontweight='bold', pad=10, color=C)

        elif chart_type == 'box':
            cols = [c for c in ([col1, col2] if col2 else [col1]) if c and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if not cols:
                cols = df.select_dtypes(include='number').columns[:5].tolist()
            data = [df[c].dropna().values for c in cols]
            if data:
                bp = ax.boxplot(data, labels=[c[:12] for c in cols], patch_artist=True,
                               medianprops=dict(color='#dc2626', linewidth=2),
                               whiskerprops=dict(color='#71717a'), capprops=dict(color='#71717a'))
                for patch in bp['boxes']:
                    patch.set_facecolor(C); patch.set_alpha(0.7)
                ax.set_title('Box Plot', fontsize=12, fontweight='bold', pad=10, color=C)

        elif chart_type == 'heatmap':
            num_df = df.select_dtypes(include='number')
            if len(num_df.columns) < 2:
                plt.close(fig); return None
            corr = num_df.corr()
            cols = corr.columns[:min(10, len(corr.columns))]
            sub  = corr.loc[cols, cols]
            im   = ax.imshow(sub.values, cmap='RdGy_r', vmin=-1, vmax=1, aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
            ax.set_xticklabels([c[:10] for c in cols], rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels([c[:10] for c in cols], fontsize=7)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    v = sub.iloc[i, j]
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                           fontsize=6, color='white' if abs(v) > 0.55 else '#3f3f46')
            ax.set_title('Correlation Heatmap', fontsize=12, fontweight='bold', pad=10, color=C)

        else:
            plt.close(fig); return None

        plt.tight_layout(pad=1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64

    except Exception:
        plt.close('all')
        return None


def chat_with_gemini(user_message, system_prompt, history):
    import time
    import google.generativeai as genai

    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return None, (
            '⚠️ **Gemini API key not set.**\n\n'
            'To enable AI chat:\n'
            '1. Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n'
            '2. Add it as `GEMINI_API_KEY` in your environment secrets\n'
            '3. Restart the server'
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        system_instruction=system_prompt,
    )
    gemini_history = []
    for msg in history[-20:]:
        role = 'user' if msg['role'] == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [msg['content']]})

    delays = [5, 15, 30]
    last_err = ''
    for attempt, wait in enumerate([0] + delays):
        if wait:
            time.sleep(wait)
        try:
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_message)
            return response.text, None
        except Exception as e:
            last_err = str(e)
            is_rate_limit = ('429' in last_err or 'quota' in last_err.lower()
                             or 'resource_exhausted' in last_err.lower()
                             or 'rate' in last_err.lower())
            is_invalid_key = ('api_key' in last_err.lower() or 'api key' in last_err.lower()
                              or 'invalid' in last_err.lower() or '400' in last_err)
            if is_invalid_key:
                return None, '⚠️ Invalid Gemini API key. Please check your `GEMINI_API_KEY` secret.'
            if not is_rate_limit:
                return None, f'⚠️ Gemini error: {last_err}'

    return None, (
        '⚠️ Gemini is currently busy (rate limit). '
        'Your free API key allows ~15 requests/minute. '
        'Please wait 60 seconds and try again.'
    )
