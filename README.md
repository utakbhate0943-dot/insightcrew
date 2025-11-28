```markdown
**Project**: National Parks — Streamlit dashboard with Azure SQL and Tableau embeds

This repository provides a Streamlit web app that connects to an Azure SQL (MSSQL) database, displays national park details, embeds a Tableau Public dashboard (hardcoded snippet in the app), and offers an NLP→SQL feature (OpenAI).

**Prerequisites (Windows)**
- **Python 3.11**: recommended to avoid binary build problems for some wheels. Install from https://www.python.org/downloads/ if needed.
- **Microsoft ODBC Driver for SQL Server**: install one of the supported drivers (17 or 18). See: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
- **Git** (optional): to clone this repo.

**Setup & Run (PowerShell)**
1. Open PowerShell and change to the project folder:

```powershell
cd C:\Users\takbh\MIS686\insightcrew
```

2. Create and activate a Python 3.11 virtual environment (example uses `.venv311`):

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate
```

3. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Copy the example env file and edit it (do not commit your secrets):

```powershell
Copy-Item .env.example .env
# Edit .env with your preferred editor and add OPENAI_API_KEY as needed
```

Example `.env` values (replace placeholders):

```
DB_SERVER=national-parks.database.windows.net
DB_DATABASE=national-parks
DB_USERNAME=national-parks
DB_PASSWORD=*****
DB_DRIVER=ODBC Driver 18 for SQL Server
OPENAI_API_KEY=*****

Note: you can also set env vars in your PowerShell session instead of a file:

```powershell
$env:DB_SERVER = "national-parks.database.windows.net"
$env:DB_DATABASE = "national-parks"
$env:DB_USERNAME = "national-parks"
$env:DB_PASSWORD = "*****"
$env:DB_DRIVER = "ODBC Driver 18 for SQL Server"
$env:OPENAI_API_KEY = "****"
```

5. Run the Streamlit app (from the project root):

```powershell
.\.venv311\Scripts\streamlit.exe run app.py
```

6. (Optional) Capture runtime logs to a file (useful if Streamlit exits unexpectedly):

```powershell
.\.venv311\Scripts\streamlit.exe run app.py 2>&1 | Tee-Object -FilePath streamlit_runtime_log.txt
```

Quick helper: once you've created `.env` and a venv, you can run the app with the provided helper script:

```powershell
.\run-dev.ps1    # loads .env, uses .venv311 (or .venv) and launches Streamlit; logs to streamlit_runtime_log.txt by default
```

**What teammates will see / use in the app**
- **Home**: lists tables and previews rows from the connected Azure SQL database.
- **Dashboards**: shows the hardcoded Tableau embed (or fallback content if the workbook blocks frames).
- **Ask (NLP)**: ask natural-language questions — the app uses OpenAI to translate to a safe SELECT query and returns results.

**Troubleshooting**
- If `pip install` fails building packages like `numpy`, ensure you're using Python 3.11 where prebuilt wheels are available.
- If you see `pyodbc`/connection errors:
  - Confirm the ODBC Driver is installed (check `ODBC Data Sources` on Windows).
  - Ensure your client IP is allowed by the Azure SQL Server firewall (Azure Portal → Networking → Add client IP).
  - Verify `DB_DRIVER` in `.env` matches an installed driver (e.g., `ODBC Driver 17 for SQL Server`).
- If Streamlit exits immediately or prints a local URL then stops:
  - Re-run with the log capture command above and share `streamlit_runtime_log.txt` for debugging.
  - If `Tee-Object` reports a file lock, pick a different logfile name or close other processes that hold the file.

**Editing the Tableau embed**
- The app currently contains a hardcoded embed HTML snippet assigned to the `TABLEAU_EMBED_HTML` variable inside `app.py`.
- To replace the embed, edit `TABLEAU_EMBED_HTML` in `app.py` with your workbook's embed code or iframe.

**Security**
- Never commit ` .env` or secret keys to source control. Use environment variables or a secret manager for production.

**Optional: Docker / Production**
- For deployment, prefer a container image that installs the ODBC driver and the Python dependencies. I can help produce a Dockerfile that installs the ODBC driver for Windows Server Core or a Linux-based driver image.

**Need help?**
- If your teammate runs into any runtime errors, ask them to paste the `streamlit_runtime_log.txt` and any browser console messages for the Tableau embed — I can interpret and fix them.

``` **Project**: National Parks — Streamlit dashboard with Azure SQL and Tableau embeds

This repository provides a Streamlit-based web app that connects to an Azure SQL (MSSQL) database, displays national park details, and can embed Tableau Public dashboards.

Local quick start

1. Install the Microsoft ODBC driver (Windows): "ODBC Driver 17 for SQL Server".
2. Copy `.env.example` to `.env` and set `DB_SERVER`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`.
3. Create a Python virtual environment and install the required packages:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

4. Launch the app:

```powershell
streamlit run app.py
```

Features
- Connects to Azure SQL using credentials from environment variables or a `.env` file.
- Lists database tables, previews rows, and offers a map view if latitude/longitude columns exist.
- Embed a Tableau Public dashboard using its embed URL.

Deployment notes
- Streamlit Cloud: pyodbc and the ODBC driver may require additional platform configuration. Consider Dockerizing the app with the ODBC driver installed.
- Azure App Service / Container Instances: create a container image that installs the ODBC driver and the app requirements.

Security
- Keep credentials in environment variables or a secret store. Do not commit `.env`.

OpenAI and Tableau
- To enable the NLP-to-SQL feature, set `OPENAI_API_KEY` in your environment (or add it to `.env`). The app will use OpenAI to translate natural language into a safe SELECT query.
- You can configure three Tableau Public embed URLs in the app sidebar or set `TABLEAU_URL_1`, `TABLEAU_URL_2`, `TABLEAU_URL_3` in the environment.

Example environment variables (PowerShell):


Next steps
- Provide your Tableau Public embed URL to show dashboards inside the app.
- If you want, I can help Dockerize the app or create an Azure deployment pipeline.
