---
name: Chat app architecture
description: How the Gemini-powered chat system works — models, views, key decisions.
---

## Structure
- App: `apps/chat/` with models, views, gemini_client, urls
- Model: `ChatMessage(user FK, role, content, chart_data TextField, created_at)`
- Template: `templates/chat.html` — full AJAX chat, no page reload
- CSS: `static/css/chat.css`

## Key decisions
- Chat requires login (`@login_required`) — messages are user-scoped
- Dataset/ML context loaded from session (`dataset_path`, `model_dir`, `ml_results`, etc.)
- Gemini model: `gemini-2.0-flash` via `google-generativeai` package
- Gemini responds with plain text; special commands appended at end:
  `[CHART:type:col1:col2]` → backend generates matplotlib chart → base64 PNG
  `[PDF_REPORT]` → frontend shows download button pointing to `/chat/pdf/`
- `chat_pdf` view simply redirects to existing `ml:report_pdf` view (no duplication)
- History: last 20 messages sent to Gemini for context window

**Why:**
- Keeping PDF generation in one place (ml/utils/report.py) avoids drift
- Special commands embedded in Gemini text are more reliable than JSON parsing
- Session holds all needed context — no DB joins needed for ML context
