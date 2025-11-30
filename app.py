import os
import urllib
import pandas as pd
import sqlalchemy
import streamlit as st
import streamlit.components.v1 as components
import base64
from pathlib import Path
import io
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

    # Create a pyodbc engine and test a lightweight connection immediately.
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        return engine
    except Exception as e:
        msg = str(e).lower()
        # If the failure looks like a missing ODBC driver, try a pure-Python pymssql fallback
        if ("can't open lib" in msg) or ("driver manager" in msg) or ("odbc driver" in msg):
            try:
                import pymssql  # type: ignore
                host = server
                port = 1433
                if "," in server:
                    parts = server.split(",", 1)
                    host = parts[0]
                    try:
                        port = int(parts[1])
                    except Exception:
                        port = 1433
                engine2 = sqlalchemy.create_engine(
                    f"mssql+pymssql://{username}:{urllib.parse.quote_plus(password)}@{host}:{port}/{database}"
                )
                # test pymssql connection
                with engine2.connect() as conn2:
                    conn2.execute(sqlalchemy.text("SELECT 1"))
                return engine2
            except Exception as e2:
                raise RuntimeError("pyodbc failed and pymssql fallback also failed; see inner exception") from e2
        # otherwise re-raise the original error for visibility
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
        return int(df.iloc[0, 0]) if not df.empty else 0
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
    iframe = f"<iframe src=\"{url}\" width=100% height=1800 frameborder=0></iframe>"
    components.html(iframe, height=1500)


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
    # --- Themed header (map image on the right) and app color palette ---
    # No uploader UI: we will look for an image file in ./assets/ and use the first match.
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    # prefer the user's file name from attachments if present, otherwise take the first image found
    preferred_names = [
        "usa-map-with-national-parks.png",
        "header.png",
        "header.jpg",
        "header.jpeg",
    ]
    header_path = None
    for name in preferred_names:
        candidate = assets_dir / name
        if candidate.exists():
            header_path = candidate
            break
    if header_path is None:
        # pick first image in assets dir
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found = list(assets_dir.glob(ext))
            if found:
                header_path = found[0]
                break

    def _render_header(title_text: str, image_path: Path | None, height: int = 260):
        # Render a header with the provided image as a full-width background and a title positioned on the right.
        if image_path is not None and image_path.exists():
            try:
                data = image_path.read_bytes()
                mime = "jpeg"
                if str(image_path).lower().endswith(".png"):
                    mime = "png"
                b64 = base64.b64encode(data).decode()
                # Use background-size:auto 100% so the image height matches the container height
                html = f"""
<div style="position:relative;width:100%;height:{height}px;overflow:hidden;border-radius:6px;margin-bottom:18px;">
    <div style="position:absolute;inset:0;background-image:url('data:image/{mime};base64,{b64}');background-size:auto 100%;background-repeat:no-repeat;background-position:right center;filter:brightness(0.55) contrast(1.05);"></div>
    <div style="position:absolute;inset:0;background:linear-gradient(90deg, rgba(6,40,21,0.04) 0%, rgba(255,245,230,0.06) 100%);mix-blend-mode:multiply;"></div>
    <div style="position:absolute;left:36px;top:50%;transform:translateY(-50%);text-align:left;padding:0 24px;">
        <h1 style="margin:0;color:#fff;font-size:40px;font-weight:800;text-shadow:0 6px 20px rgba(0,0,0,0.6);font-family:Helvetica,Arial,sans-serif;">{title_text}</h1>
    </div>
</div>
"""
                components.html(html, height=height + 8, scrolling=False)
                return
            except Exception:
                pass

        # fallback plain title
        st.markdown(f"<h1 style='color:#0b4d2e'>{title_text}</h1>", unsafe_allow_html=True)

    # Apply a National Parks–style color palette via lightweight CSS

    st.markdown(
            """
    <style>
    /* National Parks palette: soft parchment background, deep greens and warm fall accents */
    [data-testid="stAppViewContainer"] { background: #f6f4ec; color: #17221a; }
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4, [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span {
      color: #17221a !important;
    }

    /* Sidebar with richer greens */
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0f5d2f,#2a6b38); color: #f5fbf5; }
    [data-testid="stSidebar"] * { color: #f5fbf5 !important; }

    /* Make sidebar headers/titles more visible */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .css-1lsmgbg {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        letter-spacing: 0.2px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.45);
    }

    /* Make sidebar labels slightly larger and bolder for clarity */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #eef7ee !important;
        font-weight: 600 !important;
    }

    /* Accent colors */
    :root {
      --np-forest: #164b2e;
      --np-moss: #4b7a54;
      --np-fall: #b75a1e;
      --np-sand: #f2e7d6;
    }

    /* Buttons */
    .stButton>button { background: linear-gradient(180deg,var(--np-moss),var(--np-forest)); color: #fff; border-radius:8px; }

    /* DataFrame / table contrast */
    table { color: #17221a !important; }
    th { background: #efe8de !important; color: #17221a !important; }
    td { color: #17221a !important; }

    /* Expander header readability */
    .streamlit-expanderHeader, .stExpanderHeader, .css-1v3fvcr { color: #17221a !important; }

    /* Links and captions */
    a, .stCaption { color: var(--np-forest) !important; }

    /* Make code blocks and SQL snippets more readable */
    .stCodeBlock, code { background:#f3efe6; color:#17221a; }

    </style>
    """,
            unsafe_allow_html=True,
        )

    _render_header("National Parks Intelligence System", header_path)

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

    # Sidebar: (removed Tableau Embed and OpenAI quick-status per user request)

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
        st.header("Ask — Analytical Questions")
        st.markdown(
            "This page lists 8 analytical questions with SQL snippet hints. "
            "Click an item to view the SQL and press `Run SQL` to execute it against the connected database. "
            "(This replaces the previous chatbot-style NLP UI.)"
        )

        max_rows = st.number_input("Max rows to display", min_value=10, max_value=2000, value=200)

        if engine is None:
            st.warning("Database not connected. Provide credentials in environment or .env.")

        # Analytical questions and SQL snippet hints provided by user
        questions = [
            {
                "id": "q1",
                "title": "Year-over-Year Total Visits Growth (Line Chart) (vw_YearOverYearVisits)",
                "sql": """
SELECT 
    year,
    SUM(recreational_visits) + SUM(non_recreational_visits)  as total_visits,
    (SUM(recreational_visits) + SUM(non_recreational_visits)) - LAG(SUM(recreational_visits) + SUM(non_recreational_visits)) OVER (ORDER BY year) as visit_change
FROM stats
GROUP BY year
ORDER BY year;
""",
            },
            {
                "id": "q2",
                "title": "Right Season to visit the park (vw_BestSeasonToVisit)",
                "sql": """
SELECT 
    p.park_code,
    p.name as park_name,
    s.type_of_season as season,
    s.[Good] as good_months,
    s.[Limited] as limited_months,
    s.[closed] as closed_months
FROM park p
JOIN seasons s ON p.park_code = s.park_code
WHERE s.[Good] IS NOT NULL;
""",
            },
            {
                "id": "q3",
                "title": "Parks by State Count (Map) (vw_ParksByStateCount)",
                "sql": """
SELECT 
    state,
    COUNT(*) as number_of_parks
FROM park
GROUP BY state
ORDER BY number_of_parks DESC;
""",
            },
            {
                "id": "q4",
                "title": "Revenue Per Visitor by Fee Structure (vw_RevenuePerVisitor)",
                "sql": """
WITH ParkRevenue AS (
    SELECT 
        fp.park_code,
        p.name as park_name,
        fp.is_entry_free,
        fp.is_parking_free,
        AVG(fp.entry_fee) as avg_entry_fee,
        AVG(fp.pass_cost) as avg_pass_cost,
        CASE 
            WHEN fp.is_entry_free = 1 THEN 'Free Entry'
            ELSE 'Paid Entry'
        END as fee_structure
    FROM feespasses fp
    JOIN park p ON fp.park_code = p.park_code
    GROUP BY fp.park_code, p.name, fp.is_entry_free, fp.is_parking_free
),
VisitorStats AS (
    SELECT 
        park_code,
        ROUND(AVG(recreational_visits) + AVG(non_recreational_visits), 2) as avg_annual_visits
    FROM stats
    GROUP BY park_code
)
SELECT 
    pr.park_code,
    pr.park_name,
    pr.fee_structure,
    pr.is_parking_free,
    ROUND(pr.avg_entry_fee, 2) as avg_entry_fee,
    pr.avg_pass_cost,
    vs.avg_annual_visits,
    CASE 
        WHEN vs.avg_annual_visits > 0 
        THEN ROUND((COALESCE(pr.avg_entry_fee, 0) * vs.avg_annual_visits) / vs.avg_annual_visits, 2)
        ELSE 0 
    END as estimated_revenue_per_visitor,
    ROUND(pr.avg_pass_cost / NULLIF(pr.avg_entry_fee, 0), 2) as pass_to_entry_ratio
FROM ParkRevenue pr
LEFT JOIN VisitorStats vs ON pr.park_code = vs.park_code
ORDER BY fee_structure, estimated_revenue_per_visitor DESC;
""",
            },
            {
                "id": "q5",
                "title": "Campground Cost-to-Capacity & Occupancy (vw_CampgroundCostCapacityAnalysis)",
                "sql": """
SELECT 
    p.name as park_name,
    p.state,
    COUNT(DISTINCT c.campgrounds_id) as num_campgrounds,
    ROUND(AVG(c.cost), 2) as avg_cost,
    ROUND(AVG(s.concessioner_camping + s.tent_overnights + s.rv_overnights), 0) as avg_camping_nights,
    ROUND(AVG(s.concessioner_camping + s.tent_overnights + s.rv_overnights) / 
          NULLIF(COUNT(DISTINCT c.campgrounds_id), 0), 0) as nights_per_campground,
    ROUND(AVG(s.concessioner_camping + s.tent_overnights + s.rv_overnights) / 
          NULLIF(AVG(c.cost), 0), 0) as efficiency_score
FROM park p
JOIN campgrounds c ON p.park_code = c.park_code
JOIN stats s ON p.park_code = s.park_code
    AND c.cost IS NOT NULL
    AND c.cost > 0
GROUP BY p.name, p.state
HAVING COUNT(DISTINCT c.campgrounds_id) > 0
ORDER BY efficiency_score DESC;
""",
            },
            {
                "id": "q6",
                "title": "Rising Stars (vw_RisingStarParks)",
                "sql": """
WITH RecentRise AS (
    SELECT 
        p.park_code,
        p.name as park_name,
        p.state,
        s.year,
        s.recreational_visits + s.non_recreational_visits as visits,
        LAG(s.recreational_visits + s.non_recreational_visits) OVER (PARTITION BY p.park_code ORDER BY s.year) as prev_year
    FROM park p
    JOIN stats s ON p.park_code = s.park_code
    WHERE s.year >= (SELECT MAX(year) - 1 FROM stats)
)
SELECT TOP 10
    park_name,
    state,
    year,
    visits as current_visits,
    prev_year as previous_visits,
    visits - prev_year as visit_change,
    ROUND(((visits - prev_year) * 100.0) / NULLIF(prev_year, 0), 2) as rise_percent
FROM RecentRise
WHERE prev_year IS NOT NULL
    AND visits > prev_year
    AND year = (SELECT MAX(year) FROM stats)
ORDER BY rise_percent DESC;
""",
            },
            {
                "id": "q7",
                "title": "State-Level Park Statistics (vw_StateLevelParkStats)",
                "sql": """
SELECT 
    p.state,
    COUNT(DISTINCT p.park_code) as number_of_parks,
    ROUND(SUM(s.recreational_visits), 2) as total_state_visits,
    ROUND(AVG(s.recreational_visits), 2) as avg_visits_per_park,
    ROUND(SUM(s.recreational_hours), 2) as total_recreational_hours,
    ROUND(AVG(s.recreational_hours), 2) as avg_hours_per_park,
    ROUND(SUM(s.recreational_hours) * 1.0 / NULLIF(SUM(s.recreational_visits), 0), 2) as avg_hours_per_visit_state,
    SUM(s.tent_overnights + s.rv_overnights + s.backcountry_overnights) as total_camping_nights
FROM park p
JOIN stats s ON p.park_code = s.park_code
GROUP BY p.state
""",
            },
            {
                "id": "q8",
                "title": "Falling Stars - Parks Needing Attention (vw_ParksNeedingAttention)",
                "sql": """
WITH RecentDecline AS (
    SELECT 
        p.park_code,
        p.name as park_name,
        p.state,
        s.year,
        s.recreational_visits,
        LAG(s.recreational_visits) OVER (PARTITION BY p.park_code ORDER BY s.year) as prev_year
    FROM park p
    JOIN stats s ON p.park_code = s.park_code
    WHERE s.year >= (SELECT MAX(year) - 2 FROM stats)
)
SELECT TOP 10
    park_name,
    state,
    year,
    recreational_visits as current_visits,
    prev_year as previous_visits,
    ABS(recreational_visits - prev_year) as visit_change,
    ROUND(((recreational_visits - prev_year) * 100.0) / NULLIF(prev_year, 0), 2) as decline_percent
FROM RecentDecline
WHERE prev_year IS NOT NULL
    AND recreational_visits < prev_year
    AND year = (SELECT MAX(year) FROM stats)
ORDER BY decline_percent ASC;
""",
            },
        ]

        # Index hint (display only)
        index_snippet = (
            "CREATE NONCLUSTERED INDEX IX_stats_ParkCode_Year\n"
            "ON dbo.stats (park_code, year)\n"
            "INCLUDE (\n"
            "    recreational_visits,\n"
            "    non_recreational_visits,\n"
            "    recreational_hours,\n"
            "    tent_overnights,\n"
            "    rv_overnights,\n"
            "    backcountry_overnights,\n"
            "    concessioner_camping\n"
            ");"
        )

        for i, q in enumerate(questions):
            with st.expander(f"{i+1}. {q['title']}"):
                # Provide an editable SQL area pre-populated with the snippet
                sql_key = f"sql_{q['id']}"
                sql_text = st.text_area("SQL (editable)", value=q["sql"], key=sql_key, height=220)

                # Offer a small reset button to restore the original snippet
                reset_col, run_col = st.columns([1, 1])
                with reset_col:
                    if st.button("Reset to snippet", key=f"reset_{q['id']}"):
                        st.session_state[sql_key] = q["sql"]
                        sql_text = q["sql"]
                with run_col:
                    if st.button("Run SQL", key=f"run_{q['id']}"):
                        if engine is None:
                            st.error("No DB connection available. Check DB credentials.")
                        else:
                            try:
                                # Use the editable SQL text when executing
                                df = run_sql(engine, sql_text)
                                st.caption("Executed SQL:")
                                st.code(sql_text, language="sql")
                                if df is None or df.empty:
                                    st.info("Query returned no rows.")
                                else:
                                    if df.shape[0] > int(max_rows):
                                        st.warning(f"Query returned {df.shape[0]} rows; showing first {max_rows}.")
                                        st.dataframe(df.head(int(max_rows)))
                                    else:
                                        st.dataframe(df)
                            except Exception as e:
                                st.error(f"Failed to execute query: {e}")

                st.write("Use the SQL snippet above as a hint. Edit the SQL as needed before running on your database.")

        with st.expander("Index hint (display only)"):
            st.code(index_snippet, language="sql")


if __name__ == "__main__":
    main()
