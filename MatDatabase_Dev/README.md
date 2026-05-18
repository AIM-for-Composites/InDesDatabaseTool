---
title: Mat Database
emoji: "🧪"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Mat Database

Streamlit application for browsing and uploading material properties data.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

- `GEMINI_API_KEY` (optional but required for PDF extraction with Gemini)
- `DB_HOST` (optional, required for database-backed browsing/upload)
- `DB_PORT` (optional, defaults to `5432`)
- `DB_NAME` (optional, required for database-backed browsing/upload)
- `DB_USER` (optional, required for database-backed browsing/upload)
- `DB_PASSWORD` (optional, required for database-backed browsing/upload)
