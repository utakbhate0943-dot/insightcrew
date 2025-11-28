import os
import urllib
import pandas as pd
import sqlalchemy
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="National Parks Intelligence", layout="wide")

# Hardcoded Tableau embed snippet (paste your full embed HTML/JS here)
TABLEAU_EMBED_HTML = """
<div class='tableauPlaceholder' id='viz1764260932811' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USNationalParksDashboard_17642532718610&#47;NPIntelligentsystem&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='USNationalParksDashboard_17642532718610&#47;NPIntelligentsystem' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USNationalParksDashboard_17642532718610&#47;NPIntelligentsystem&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1764260932811');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1400px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='1550px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1400px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='1550px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='2450px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
"""


def get_db_config():
    # Prefer environment variables (or .env via load_dotenv()),
    # but fall back to Streamlit secrets when available (deployed on Streamlit Cloud).
    def _get(key, default=None):
        v = os.getenv(key)
        if v:
            return v
        try:
            return st.secrets.get(key, default)
        except Exception:
            return default

    server = _get("DB_SERVER")
    database = _get("DB_DATABASE")
    username = _get("DB_USERNAME")
    password = _get("DB_PASSWORD")
    driver = _get("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    return server, database, username, password, driver


@st.cache_resource
def make_engine(server, database, username, password, driver):
    if not (server and database and username and password):
        return None
    # First try using pyodbc (requires system ODBC driver e.g. msodbcsql17/18)
    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    quoted = urllib.parse.quote_plus(conn_str)
    try:
        engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")
        return engine
    except Exception as e:
        # Common failure: ODBC driver not installed on the host ("Can't open lib ... file not found").
        msg = str(e)
        # If the error looks like a missing driver, try a pure-Python fallback (pymssql / FreeTDS)
        if "Can't open lib" in msg or "Driver" in msg or "data source name" in msg.lower():
            try:
                # Import here so requirement is optional for local dev until installed
                import pymssql  # type: ignore
                # Build a pymssql connection URL. Use default port 1433 if none specified.
                host = server
                port = 1433
                if "," in server:
                    # server may be in the form 'host,port'
                    parts = server.split(",", 1)
                    host = parts[0]
                    try:
                        port = int(parts[1])
                    except Exception:
                        port = 1433
                engine = sqlalchemy.create_engine(f"mssql+pymssql://{username}:{urllib.parse.quote_plus(password)}@{host}:{port}/{database}")
                return engine
            except Exception:
                # re-raise original error to preserve context (but avoid exposing credentials)
                raise RuntimeError("pyodbc failed and pymssql fallback also failed; see inner exceptions for details") from e
        # If it's another kind of error, raise it
        raise


def test_db_connection(engine):
    """Attempt a minimal DB connection and return (ok, error_message).
    This does not log credentials; any returned error may help diagnose network/driver issues.
    """
    if engine is None:
        return False, "No engine (missing credentials)"
    try:
        with engine.connect() as conn:
            # lightweight check
            conn.execute(sqlalchemy.text("SELECT 1"))
        return True, None
    except Exception as e:
        # return stringified error for diagnostics (do not include full connection string)
        return False, str(e)


@st.cache_data(ttl=60)
def list_tables(_engine):
    try:
        sql = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
        df = pd.read_sql(sql, _engine)
        df["full_name"] = df["TABLE_SCHEMA"] + "." + df["TABLE_NAME"]
        return df["full_name"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=30)
def read_table(_engine, table_name, limit=500, where=None):
    try:
        where_clause = f" WHERE {where}" if where and where.strip() else ""
        query = f"SELECT TOP ({limit}) * FROM {table_name}{where_clause}"
        return pd.read_sql(query, _engine)
    except Exception as e:
        # Do not crash the app; return empty df and surface the error where called
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_table_schema(_engine, table_name):
    try:
        if '.' in table_name:
            schema, tbl = table_name.split('.', 1)
        else:
            schema, tbl = 'dbo', table_name
        sql = (
            "SELECT COLUMN_NAME, DATA_TYPE"
            " FROM INFORMATION_SCHEMA.COLUMNS"
            " WHERE TABLE_SCHEMA = '" + schema + "' AND TABLE_NAME = '" + tbl + "'"
            " ORDER BY ORDINAL_POSITION"
        )
        return pd.read_sql(sql, _engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_table_row_count(_engine, table_name):
    try:
        sql = f"SELECT COUNT(*) as cnt FROM {table_name}"
        df = pd.read_sql(sql, _engine)
        return int(df.iloc[0, 0])
    except Exception:
        return None


def infer_latlon_cols(df):
    if df.empty:
        return None, None
    lat_candidates = [c for c in df.columns if c.lower() in ("latitude", "lat")]
    lon_candidates = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng", "long")]
    lat = lat_candidates[0] if lat_candidates else None
    lon = lon_candidates[0] if lon_candidates else None
    return lat, lon


def render_tableau_embed(url):
    if not url:
        st.info("No Tableau URL provided.")
        return
    # If the input looks like raw embed HTML (starts with '<'), render it directly.
    trimmed = url.strip()
    if trimmed.startswith("<"):
        # The user pasted the full Tableau embed snippet (HTML + script).
        # Render raw HTML safely inside Streamlit component.
        components.html(trimmed, height=900, scrolling=True)
        return

    # Otherwise assume it's a direct URL and render as iframe.
    iframe = f"<iframe src=\"{url}\" width=100% height=1400 frameborder=0></iframe>"
    components.html(iframe, height=820)


def nlp_to_sql_openai(prompt_text, engine, max_tokens=256, model="gpt-4o-mini"):
    # This function expects OPENAI_API_KEY set in env. Uses OpenAI to produce a SQL SELECT only.
    # Prefer the new OpenAI client if available (openai>=1.0.0). Fall back to the older
    # openai.ChatCompletion interface for compatibility with older installs.
    schema_hint = "\n".join(list_tables(engine)[:10]) if engine is not None else "(schema not available)"

    system_prompt = (
        "You are a helpful assistant that translates natural language questions into SQL SELECT queries. "
        "Only return a single syntactically valid T-SQL SELECT statement. Do NOT return any non-SELECT statements (no INSERT/UPDATE/DELETE/DROP/etc). "
        "If the question is ambiguous, ask follow-up for clarification. Always constrain results (use TOP) to avoid very large scans. "
        "Use the following available tables (examples):\n" + schema_hint
    )

    sql = None
    # Try new client API first
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        # response structure may vary; try common access patterns
        try:
            sql = resp.choices[0].message.content.strip()
        except Exception:
            try:
                sql = resp.choices[0]["message"]["content"].strip()
            except Exception:
                sql = getattr(resp.choices[0], 'text', None)
                if sql:
                    sql = sql.strip()
    except Exception:
        # Fall back to older openai package interface
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if openai.api_key is None:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            try:
                sql = resp.choices[0].message.content.strip()
            except Exception:
                sql = getattr(resp.choices[0], 'text', None)
                if sql:
                    sql = sql.strip()
        except Exception as e:
            raise RuntimeError(
                "OpenAI request failed. Either set OPENAI_API_KEY or install a compatible openai package (e.g. openai>=1.0.0), or pin to openai==0.28 if you prefer the old API. "
                f"Inner error: {e}"
            )

    if not sql:
        raise RuntimeError("No SQL returned from OpenAI model")
    # Simple safety: disallow suspicious keywords
    lowered = sql.lower()
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate ", "exec ", "sp_"]
    if any(k in lowered for k in forbidden):
        raise RuntimeError("Generated SQL contains forbidden statements")

    # Ensure SELECT
    if not lowered.startswith("select"):
        raise RuntimeError("Generated SQL does not start with SELECT")

    # Ensure TOP exists to limit result size
    if "top" not in lowered:
        # naive: add TOP 500
        sql = sql.replace("select", "SELECT TOP (500)", 1)

    return sql


def run_sql(engine, sql):
    try:
        df = pd.read_sql(sql, engine)
        return df
    except Exception as e:
        raise


def main():
    st.title("National Parks Intelligence System")

    # Sidebar: DB + Tableau + OpenAI config
    st.sidebar.header("Connection & Configuration")
    server, database, username, password, driver = get_db_config()
    st.sidebar.text("DB Server:")
    st.sidebar.caption(server or "(not set)")

    engine = make_engine(server, database, username, password, driver)

    # Diagnostic: show where credentials are coming from (env / secrets / missing)
    def _source_of(key):
        if os.getenv(key):
            return "env"
        try:
            if key in st.secrets and st.secrets.get(key) is not None:
                return "secrets"
        except Exception:
            pass
        return "(missing)"

    cred_sources = {
        "server": _source_of("DB_SERVER"),
        "database": _source_of("DB_DATABASE"),
        "username": _source_of("DB_USERNAME"),
        "password": _source_of("DB_PASSWORD"),
    }

    # Masked display for diagnostics (do not show password)
    st.sidebar.markdown("**DB credential sources**")
    st.sidebar.write(f"Server: {cred_sources['server']}")
    st.sidebar.write(f"Database: {cred_sources['database']}")
    st.sidebar.write(f"Username: {cred_sources['username']}")
    st.sidebar.write(f"Password: {cred_sources['password']} (hidden)")

    # Test connection and show any error for debugging
    ok, err = test_db_connection(engine)
    if not ok and err:
        st.sidebar.error("DB connection test failed — see message below")
        st.sidebar.code(err)
    elif ok:
        st.sidebar.success("DB connection test: OK")

    # Single hardcoded Tableau dashboard; no UI inputs for embeds
    st.sidebar.markdown("---")
    st.sidebar.header("Tableau Embed")
    st.sidebar.write("This app displays a single embedded Tableau dashboard (hardcoded).")

    st.sidebar.markdown("---")
    st.sidebar.header("OpenAI")
    if os.getenv("OPENAI_API_KEY"):
        st.sidebar.success("OpenAI key loaded")
    else:
        st.sidebar.info("Set OPENAI_API_KEY in env to enable NLP-to-SQL")

    page = st.sidebar.radio("Page", ["Home", "Dashboards", "Ask (NLP)"])

    if page == "Home":
        st.header("Overview")
        st.markdown("This portal shows national park details and dashboards. Use the Dashboards page for Tableau embeds and Ask page to query the database using natural language.")

        if engine is None:
            st.warning("Database not connected. Provide credentials in environment or .env.")
        else:
            # Try to show a summary of a parks table if present
            tables = list_tables(engine)
            st.subheader("Available tables")
            if not tables:
                st.write("(no tables found or cannot connect)")
            else:
                # let the user pick a table to preview
                selected = st.selectbox("Choose a table to preview", options=tables)
                if selected:
                    st.subheader(f"Preview: {selected}")

                    # Show schema and row count
                    schema_df = get_table_schema(engine, selected)
                    row_count = get_table_row_count(engine, selected)
                    col1, col2 = st.columns([3,1])
                    with col1:
                        if not schema_df.empty:
                            st.markdown("**Columns (name : type)**")
                            st.write(schema_df.set_index('COLUMN_NAME')['DATA_TYPE'].to_dict())
                        else:
                            st.info("Could not retrieve column schema")
                    with col2:
                        st.markdown("**Row count**")
                        st.write(row_count if row_count is not None else "unknown")

                    # Preview options (no raw WHERE clause for safety)
                    st.markdown("---")
                    st.markdown("**Preview options**")
                    max_rows = st.number_input("Limit rows", min_value=10, max_value=5000, value=50, step=10)

                    # Load preview (cached)
                    df = read_table(engine, selected, limit=int(max_rows), where=None)
                    if df is None or df.empty:
                        st.info("No rows available for this table or failed to read.")
                    else:
                        st.dataframe(df)
                        lat, lon = infer_latlon_cols(df)
                        if lat and lon:
                            # normalize and coerce to numeric to avoid map errors
                            map_df = df.rename(columns={lat: "latitude", lon: "longitude"})[["latitude", "longitude"]].copy()
                            map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors='coerce')
                            map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors='coerce')
                            map_df = map_df.dropna()
                            if not map_df.empty:
                                st.map(map_df)
                            else:
                                st.info("No valid numeric latitude/longitude values to map for this preview.")

                        # Download as CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download preview as CSV", data=csv, file_name=f"{selected.replace('.','_')}_preview.csv", mime='text/csv')

    elif page == "Dashboards":
        st.header("Embedded Dashboard")
        st.markdown("This page displays the single hardcoded Tableau dashboard.")
        # Render the hardcoded Tableau embed HTML/snippet
        try:
            # Show the embed (the snippet includes the script that loads Tableau JS)
            components.html(TABLEAU_EMBED_HTML, height=1000, scrolling=True)
        except Exception as e:
            st.error(f"Failed to render embedded dashboard: {e}")

    else:
        st.header("Ask the Database (NLP)")
        st.markdown("Ask natural language questions and (if OpenAI key provided) the app will attempt to translate them to a safe SELECT query and return results.")

        # Example prompts users can click to populate the question box
        if "nlp_question" not in st.session_state:
            st.session_state["nlp_question"] = "List top 10 parks by area"
        if "run_now" not in st.session_state:
            st.session_state["run_now"] = False

        examples = [
            ("Summary of schema", "Give me a short summary of the tables and what they contain"),
            ("Parks table columns", "What columns does the Parks table have?"),
            ("Top 10 by visitors", "Show the top 10 parks by annual_visitors"),
            ("Parks in California", "List parks in California with area greater than 100000"),
            ("Yellowstone visitors 2018-2022", "Get the visitor counts for Yellowstone between 2018 and 2022"),
            ("Keyword search", "Find parks mentioning 'bear' in description"),
        ]

        with st.expander("Example questions (click to use)"):
            for i, (label, prompt) in enumerate(examples):
                col1, col2 = st.columns([5,1])
                with col1:
                    st.write(f"**{label}** — {prompt}")
                with col2:
                    if st.button("Use", key=f"ex_use_{i}"):
                        st.session_state["nlp_question"] = prompt
                    if st.button("Run", key=f"ex_run_{i}"):
                        st.session_state["nlp_question"] = prompt
                        st.session_state["run_now"] = True

        question = st.text_area("Question", value=st.session_state.get("nlp_question", ""), key="nlp_question", height=120)
        max_rows = st.number_input("Max rows to return", min_value=10, max_value=2000, value=200)
        run = st.button("Run")

        # If an example's Run button was pressed, set local run flag and clear the session marker
        if st.session_state.get("run_now"):
            run = True
            st.session_state["run_now"] = False

        if run:
            if engine is None:
                st.error("No DB connection available. Check DB credentials in environment or .env.")
            else:
                # If OpenAI key present, use model; otherwise fall back to a simple keyword search
                if os.getenv("OPENAI_API_KEY"):
                    try:
                        sql = nlp_to_sql_openai(question, engine)
                        st.code(sql, language="sql")
                        df = run_sql(engine, sql)
                        if df.shape[0] > max_rows:
                            st.warning(f"Query returned {df.shape[0]} rows, truncating to {max_rows}")
                            st.dataframe(df.head(max_rows))
                        else:
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Failed to generate/execute SQL: {e}")
                else:
                    # naive fallback: search parks table for keywords
                    tables = list_tables(engine)
                    candidate = next((t for t in tables if "park" in t.lower()), None)
                    if not candidate:
                        st.error("No parks table found and OpenAI key not set.")
                    else:
                        # build a simple LIKE-based query across textual columns
                        df_sample = read_table(engine, candidate, limit=10)
                        text_cols = [c for c in df_sample.columns if df_sample[c].dtype == object]
                        if not text_cols:
                            st.error("No text columns available for fallback search")
                        else:
                            # Escape single quotes for T-SQL and build WHERE clause
                            escaped = question.replace("'", "''")
                            where = " OR ".join([f"[{c}] LIKE '%{escaped}%'" for c in text_cols])
                            sql = f"SELECT TOP ({max_rows}) * FROM {candidate} WHERE {where}"
                            st.code(sql, language="sql")
                            df = run_sql(engine, sql)
                            st.dataframe(df)


if __name__ == "__main__":
    main()
