```markdown
**Project**: National Parks — Streamlit dashboard with Azure SQL and Tableau embeds

This repository provides a Streamlit web app that connects to an Azure SQL (MSSQL) database, displays national park details, and embeds a Tableau Public dashboard (hardcoded snippet in the app).

**Prerequisites (Windows)**
- **Python 3.11**: recommended to avoid binary build problems for some wheels. Install from https://www.python.org/downloads/ if needed.
- **Microsoft ODBC Driver for SQL Server**: install one of the supported drivers (17 or 18). See: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
- **Git** (optional): to clone this repo.

**Setup & Run (PowerShell)**
1. Open PowerShell and change to the project folder:

```powershell
# National Parks — Streamlit Explorer

This repository is a Streamlit app for exploring a National Parks MSSQL database and embedding a Tableau dashboard. The project is focused on local development and data exploration.

Key current decisions

- The app no longer depends on OpenAI or any NLP service.
- Analytical questions are loaded from `questions.json` in the project root (editable JSON list).
- Preview mapping functionality was removed — previews will not attempt to render maps.
- The Gallery view reads images from the `park` table only.

Prerequisites (Windows)

- Python 3.11 (recommended)
- Microsoft ODBC Driver for SQL Server (17 or 18)

Quick start (PowerShell)

```powershell
cd C:\Users\takbh\MIS686\insightcrew
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit `.env` and set DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD
.\.venv311\Scripts\streamlit.exe run app.py
```

Notes

- Database credentials can be provided via environment variables, a `.env` file, or Streamlit `secrets`.
- Edit `questions.json` to change the prebuilt SQL snippets shown on the Ask page.

Troubleshooting

- If connection errors occur, verify the ODBC driver installation and Azure SQL firewall settings.
- If Streamlit exits immediately, capture logs:

```powershell
.\.venv311\Scripts\streamlit.exe run app.py 2>&1 | Tee-Object -FilePath streamlit_runtime_log.txt
```

If you'd like, I can:

- Run the app locally and confirm the Ask/Gallery pages behave as expected.
- Remove debug helpers and assistant-added comments from `app.py`.
- Create a Dockerfile for deployment.

---
Updated: December 2025
py -3.11 -m venv .venv311
