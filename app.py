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

# Hardcoded Tableau embed snippet
TABLEAU_EMBED_HTML = """
<div class='tableauPlaceholder' id='viz1764637863337' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USNationalParksDashboard_17642532718610&#47;ExplorerGuide&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='USNationalParksDashboard_17642532718610&#47;ExplorerGuide' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USNationalParksDashboard_17642532718610&#47;ExplorerGuide&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1764637863337');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='900px';vizElement.style.height='1250px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='900px';vizElement.style.height='1250px';} else { vizElement.style.width='100%';vizElement.style.height='2450px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
"""

def get_db_config():
    """Resolve DB credentials from environment variables or Streamlit secrets.

    Returns:
        tuple: (server, database, username, password, driver)
    """
    def _get(key: str, default=None):
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
    
    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    quoted = urllib.parse.quote_plus(conn_str)

    
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        return engine
    except Exception as e:
        last_err = str(e)
        try:
            st.session_state['db_connect_error'] = last_err
        except Exception:
            pass

        
        try:
            alt_conn_str = (
                f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
                "Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
            )
            alt_quoted = urllib.parse.quote_plus(alt_conn_str)
            engine_alt = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={alt_quoted}")
            with engine_alt.connect() as conn_alt:
                conn_alt.execute(sqlalchemy.text("SELECT 1"))
            return engine_alt
        except Exception:
                
            try:
                alt_conn_str2 = (
                    f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
                    "Encrypt=no;TrustServerCertificate=yes;Connection Timeout=30;"
                )
                alt_quoted2 = urllib.parse.quote_plus(alt_conn_str2)
                engine_alt2 = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={alt_quoted2}")
                with engine_alt2.connect() as conn_alt2:
                    conn_alt2.execute(sqlalchemy.text("SELECT 1"))
                return engine_alt2
            except Exception:
                
                msg = last_err.lower()
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
                    
                    with engine2.connect() as conn2:
                        conn2.execute(sqlalchemy.text("SELECT 1"))
                    try:
                        st.session_state['db_connect_error'] = None
                    except Exception:
                        pass
                    return engine2
                except Exception:
                    
                    try:
                        st.session_state['db_connect_error'] = last_err
                    except Exception:
                        pass
                    return None


def test_db_connection(engine):
    """
    Attempt a minimal DB connection and return (ok, error_message).
    This check executes a lightweight "SELECT 1" to verify connectivity and
    surface a human-readable error (without exposing credentials) for diagnostics.
    Returns:
      (True, None) on success
      (False, error_message) on failure
    """
    if engine is None:
        return False, "No engine (missing credentials)"
    try:
        with engine.connect() as conn:
    
            conn.execute(sqlalchemy.text("SELECT 1"))
        return True, None
    except Exception as e:
        
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


@st.cache_data(ttl=60)
def list_views(_engine):
    
    try:
        sql = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS"
        df = pd.read_sql(sql, _engine)
        df["full_name"] = df["TABLE_SCHEMA"] + "." + df["TABLE_NAME"]
        
        try:
            df = df[df["full_name"].str.lower() != "sys.database_firewall_rules"]
        except Exception:
            pass
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
        
        return pd.DataFrame()


def _reset_sql_in_session(key, value):
    """Callback used by Reset buttons to update `st.session_state` before widgets render."""
    try:
        st.session_state[key] = value
    except Exception:
        
        pass

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

def get_object_ddl(_engine, full_name: str) -> str | None:
    """
    Attempt to retrieve the DDL/definition for a view or the CREATE TABLE
    statement for a table. For views (or other programmable objects) we try
    `OBJECT_DEFINITION(OBJECT_ID('schema.name'))`. If that is not available
    (e.g. for plain tables), we fall back to constructing a basic CREATE
    TABLE statement using INFORMATION_SCHEMA metadata.
    Returns the SQL string or None if not available.
    """
    try:
        if not full_name:
            return None
        if '.' in full_name:
            schema, name = full_name.split('.', 1)
        else:
            schema, name = 'dbo', full_name

        
        try:
            sql = f"SELECT OBJECT_DEFINITION(OBJECT_ID('{schema}.{name}')) AS definition"
            df = pd.read_sql(sql, _engine)
            if not df.empty and pd.notna(df.iloc[0, 0]):
                return df.iloc[0, 0]
        except Exception:
            
            pass

        
        cols = get_table_schema(_engine, full_name)
        if cols is None or cols.empty:
            return None
        lines = []
        for _, r in cols.iterrows():
            col = r.get('COLUMN_NAME')
            dt = r.get('DATA_TYPE')
            if pd.isna(col) or pd.isna(dt):
                continue
            lines.append(f"    [{col}] {dt}")
        ddl = f"CREATE TABLE [{schema}].[{name}] (\n" + ",\n".join(lines) + "\n);"
        return ddl
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_table_row_count(_engine, table_name):
    
    try:
        sql = f"SELECT COUNT(*) as cnt FROM {table_name}"
        df = pd.read_sql(sql, _engine)
        return int(df.iloc[0, 0]) if not df.empty else 0
    except Exception:
        return None



def render_tableau_embed(url_or_html: str, height: int = 1400):
    # Render a Tableau embed snippet or a direct Tableau URL.
    if not url_or_html:
        st.info("No Tableau URL provided.")
        return

    trimmed = url_or_html.strip()
    if trimmed.startswith("<"):
        try:
            import re

            # Replace assignments like vizElement.style.width='900px'; or vizElement.style.width="900px";
            html_safe = re.sub(r"vizElement\.style\.width\s*=\s*[^;]+;", "vizElement.style.width='100%';", trimmed, flags=re.IGNORECASE)
            # Also replace any hard-coded width attributes on object/embed tags
            html_safe = re.sub(r"width=\"\d+px\"", 'width="100%"', html_safe, flags=re.IGNORECASE)
            html_safe = re.sub(r"width='\d+px'", "width='100%'", html_safe, flags=re.IGNORECASE)

            # Ensure the outer placeholder div will let the embedded object size to container
            # by forcing object/tableauViz elements to be 100% width via inline CSS
            style_inject = "<style>.tableauViz, .tableauPlaceholder object, .tableauPlaceholder embed { width:100% !important; max-width:100% !important; }</style>"
            if style_inject not in html_safe:
                html_safe = style_inject + html_safe

            components.html(html_safe, height=height, scrolling=True)
        except Exception:
            # On any failure, fall back to rendering the original snippet
            components.html(trimmed, height=height, scrolling=True)
        return

    # Otherwise assume it's a direct URL and render as a responsive iframe.
    iframe = f'<div style="width:100%"><iframe src="{url_or_html}" style="width:100%;height:100%;border:0;" loading="lazy"></iframe></div>'
    components.html(iframe, height=height, scrolling=True)

def run_sql(engine, sql):
    # Execute a SQL query and return a pandas DataFrame. Exceptions are
    try:
        df = pd.read_sql(sql, engine)
        return df
    except Exception as e:
        raise


def sanitize_sql_for_run(sql_text: str, max_limit: int = 500) -> str:
    """
    Basic static validation / sanitization for user-provided SQL before execution.
    - Blocks forbidden keywords (INSERT/UPDATE/DELETE/DROP/etc).
    - If the query is a plain SELECT without TOP/GROUP BY/WITH, injects a TOP(max_limit)
      to avoid accidental large scans.

    This is intentionally conservative and performs textual heuristics only.
    """
    if not sql_text or not isinstance(sql_text, str):
        return sql_text

    lowered = sql_text.lower()
    # Block potentially destructive keywords
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate ", "exec ", "sp_"]
    for k in forbidden:
        if k in lowered:
            raise RuntimeError(f"Blocked forbidden keyword in SQL: {k.strip()}")

    # If the SQL starts with WITH (CTE) or contains GROUP BY/UNION/LIMIT/TOP, avoid modifying
    starts_with_with = lowered.lstrip().startswith("with ")
    contains_group = "group by" in lowered
    contains_union = "union" in lowered
    contains_top = "select top" in lowered or "top (" in lowered
    contains_limit = " limit " in lowered

    if starts_with_with or contains_group or contains_union or contains_top or contains_limit:
        return sql_text

    # If it's a plain SELECT without TOP, add a TOP(...) after the first SELECT
    stripped = sql_text.lstrip()
    if stripped.lower().startswith("select") and not contains_top:
        # insert TOP after the first SELECT token
        # preserve original casing for 'SELECT'
        prefix = sql_text[: sql_text.lower().find("select")]
        rest = sql_text[sql_text.lower().find("select") + len("select"):]
        return prefix + "SELECT TOP (" + str(int(max_limit)) + ")" + rest

    return sql_text


def render_sample_viz(qid: str, df: pd.DataFrame):
    """Render a small, safe sample visualization for known analytical question outputs.
    Uses Streamlit built-in charts so no extra plotting deps are required.
    """
    if df is None or df.empty:
        return
    try:
        # This function contains hard-coded small visualizations keyed by
        # question id (qid). The visuals are intentionally simple and use only Streamlit's built-in charts so the app remains lightweight.
        # Q1: Year-over-Year total visits
        if qid == "q1" and "year" in df.columns:
            d = df.copy()
            if "total_visits" in d.columns:
                d["year"] = pd.to_numeric(d["year"], errors="coerce")
                d = d.dropna(subset=["year"]).sort_values("year")
                if not d.empty:
                    st.subheader("Year-over-Year total visits")
                    st.line_chart(d.set_index("year")["total_visits"], use_container_width=True)
                    return

        # Q2: Best season — counts per season
        if qid == "q2" and "season" in df.columns:
            st.subheader("Parks by Season (sample)")
            counts = df["season"].value_counts().rename_axis("season").reset_index(name="count")
            st.bar_chart(counts.set_index("season")["count"], use_container_width=True)
            return

        # Q3: Parks by state count
        if qid == "q3" and "state" in df.columns and ("number_of_parks" in df.columns or "count" in df.columns):
            st.subheader("Parks by state (top 20)")
            key = "number_of_parks" if "number_of_parks" in df.columns else "count"
            top = df[["state", key]].groupby("state").sum().sort_values(key, ascending=False).head(20)
            st.bar_chart(top[key], use_container_width=True)
            return

        # Q4: Revenue per visitor
        if qid == "q4" and "park_name" in df.columns and "estimated_revenue_per_visitor" in df.columns:
            st.subheader("Estimated revenue per visitor (sample)")
            sample = df[["park_name", "estimated_revenue_per_visitor"]].dropna()
            sample = sample.sort_values("estimated_revenue_per_visitor", ascending=False).head(15)
            st.bar_chart(sample.set_index("park_name")["estimated_revenue_per_visitor"], use_container_width=True)
            return

        # Q5: Campground cost vs avg camping nights (scatter via Vega-Lite)
        if qid == "q5" and "avg_cost" in df.columns and "avg_camping_nights" in df.columns:
            st.subheader("Campground cost vs avg camping nights")
            spec = {
                "mark": "point",
                "encoding": {
                    "x": {"field": "avg_cost", "type": "quantitative", "title": "Avg Cost"},
                    "y": {"field": "avg_camping_nights", "type": "quantitative", "title": "Avg camping nights"},
                    "tooltip": [{"field": "park_name", "type": "nominal"}] if "park_name" in df.columns else []
                }
            }
            st.vega_lite_chart(df.dropna(subset=["avg_cost", "avg_camping_nights"]).head(500), spec, use_container_width=True)
            return

        # Q6: Rising stars
        if qid == "q6" and "rise_percent" in df.columns:
            st.subheader("Top rising parks")
            sample = df[["park_name", "rise_percent"]].dropna().sort_values("rise_percent", ascending=False).head(15)
            st.bar_chart(sample.set_index("park_name")["rise_percent"], use_container_width=True)
            return

        # Q7: State-level stats
        if qid == "q7" and "state" in df.columns and "total_state_visits" in df.columns:
            st.subheader("State total visits (sample)")
            top = df[["state", "total_state_visits"]].groupby("state").sum().sort_values("total_state_visits", ascending=False).head(20)
            st.bar_chart(top["total_state_visits"], use_container_width=True) 
            return

        # Q8: Falling stars
        if qid == "q8" and ("decline_percent" in df.columns or "visit_change" in df.columns):
            st.subheader("Parks needing attention")
            key = "decline_percent" if "decline_percent" in df.columns else "visit_change"
            sample = df[["park_name", key]].dropna().sort_values(key).head(15)
            st.bar_chart(sample.set_index("park_name")[key], use_container_width=True)
            return

    except Exception:
        # Keep visualization best-effort — if anything fails, do not break the app
        return


def main():
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
        <h1 style="margin:0;color:#000;font-size:36px;line-height:1;font-weight:900;text-shadow:none;font-family:'TW Cen MT Condensed Extra Bold','TW Cen MT Condensed','Arial Black',Arial,sans-serif;">{title_text}</h1>
    </div>
</div>
"""
                components.html(html, height=height + 8, scrolling=False)
                return
            except Exception:
                pass

        # fallback plain title (uses requested condensed font if available)
        st.markdown(
            f"<h1 style=\"margin:0;color:#000;font-size:36px;line-height:1;font-family:'TW Cen MT Condensed Extra Bold','TW Cen MT Condensed','Arial Black',Arial,sans-serif;font-weight:900;\">{title_text}</h1>",
            unsafe_allow_html=True,
        )

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

    /* Make sidebar headers/titles more visible and larger */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .css-1lsmgbg {
        color: #ffffff !important;
        font-size: 22px !important; /* larger for visibility */
        font-weight: 900 !important;
        letter-spacing: 0.2px;
        line-height: 1.05 !important;
        text-shadow: 0 3px 10px rgba(0,0,0,0.55);
    }

    /* Make sidebar labels slightly larger, bolder, and clearer */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stText {
        color: #f3fbf3 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }

    /* Emphasize small status badges and code blocks in the sidebar */
    [data-testid="stSidebar"] .stCode, [data-testid="stSidebar"] code { font-size:12px !important; color:#f7fff7 !important; }

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
    th { background: #FFFFFF !important; color: #17221a !important; }
    td { color: #17221a !important; }

    /* Expander header readability */
    .streamlit-expanderHeader, .stExpanderHeader, .css-1v3fvcr { color: #17221a !important; }

    /* Links and captions */
    a, .stCaption { color: var(--np-forest) !important; }

    /* Make code blocks and SQL snippets more readable */
    .stCodeBlock, code { background:#f3efe6; color:#17221a; }

    /* UI spacing and preview panel styling */
    .preview-panel { padding: 14px 18px !important; background: #fffdf6 !important; border-radius:10px !important; border:1px solid #e9e0d2 !important; box-shadow: 0 4px 10px rgba(0,0,0,0.03) !important; margin-bottom:14px !important; }
    .preview-title { color: var(--np-forest) !important; font-weight:700; font-size:16px; margin-bottom:8px; }

    /* Make expanders a bit roomier */
    .stAppViewContainer .stExpanderContent, .stAppViewContainer .streamlit-expanderContent {
        padding: 18px !important;
    }

    /* Inputs: increase padding and subtle border for better contrast */
    textarea, input[type="text"], .stNumberInput input {
        padding: 10px !important; border-radius:6px !important; border:1px solid #e6dfd1 !important; background: #fffdf8 !important; color: #17221a !important;
    }

    /* Buttons: consistent sizing and higher contrast */
    .stButton>button { padding: 8px 14px !important; font-weight:700 !important; box-shadow:none !important; }

    /* DataFrame / table tile padding */
    .stAppViewContainer .stDataFrame, .stAppViewContainer .stTable, .stAppViewContainer .element-container {
        padding: 12px !important;
    }

    /* Make the preview labels slightly darker for readability */
    .stMarkdown b, .stMarkdown strong { color: #143723 !important; }

    /* DB status badge and credential styling */
    .db-badge { display:inline-block; padding:6px 12px; border-radius:10px; color:#fff; font-weight:800; font-size:14px; margin-bottom:8px; }
    .db-badge-success { background: linear-gradient(90deg,#38b24a,#0f7a2e); box-shadow: 0 4px 12px rgba(15,122,46,0.18); }
    .db-badge-fail { background: linear-gradient(90deg,#e05d4f,#b22b1a); box-shadow: 0 4px 12px rgba(178,43,26,0.18); }
    .db-cred-line { padding:6px 8px; border-radius:6px; background: rgba(255,255,255,0.04); color:#f7fff7; margin-bottom:6px; font-weight:700; }
    .db-cred-label { color: #e6f6e9; font-weight:800; margin-right:6px; }
    .db-err-block { background:#2b0f0f; color:#ffecec; padding:8px; border-radius:6px; font-size:13px; }

    </style>
    """,
            unsafe_allow_html=True,
        )

    _render_header("National Parks Intelligence System", header_path)
    # Icon-only page selector placed directly under the header.We keep internal page identifiers (Home/Dashboards/Gallery/Ask)
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Home"

    pages = ["Home", "Dashboards", "Gallery", "Ask"]
    # compact horizontal radio with no visible label (looks like the screenshot)
    try:
        current_index = pages.index(st.session_state.get("selected_page", "Home"))
    except Exception:
        current_index = 0

    # Use a non-empty label for accessibility, but hide it visually
    sel = st.radio("Page selector", pages, index=current_index, horizontal=True, key="page_radio", label_visibility="collapsed")
    # keep a stable alias for other code
    st.session_state["selected_page"] = sel
    page = sel

    # Sidebar: DB config
    st.sidebar.header("Connection & Configuration")
    server, database, username, password, driver = get_db_config()
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
    # Render credential lines using small styled blocks for better contrast
    # Show actual credential values where available (password kept hidden)
    display_server = server if server else "(not set)"
    display_database = database if database else "(not set)"
    display_username = username if username else "(not set)"
    display_password = "(hidden)" if password else "(not set)"

    creds_html = (
        f"<div class='db-cred-line'><span class='db-cred-label'>Server:</span> {display_server}</div>"
        f"<div class='db-cred-line'><span class='db-cred-label'>Database:</span> {display_database}</div>"
        f"<div class='db-cred-line'><span class='db-cred-label'>Username:</span> {display_username}</div>"
        f"<div class='db-cred-line'><span class='db-cred-label'>Password:</span> {display_password}</div>"
    )
    st.sidebar.markdown(creds_html, unsafe_allow_html=True)

    # Test connection and show any error for debugging
    # If make_engine returned None but stored a connection error, surface that message to the user
    err = None
    ok = False
    if engine is None:
        conn_err = None
        try:
            conn_err = st.session_state.get('db_connect_error')
        except Exception:
            conn_err = None
        if conn_err:
            err = conn_err
        else:
            err = "No engine (missing credentials)"
        ok = False
    else:
        ok, err = test_db_connection(engine)

    # Show a prominent colored badge for the connection status
    if not ok and err:
        st.sidebar.markdown("<div class='db-badge db-badge-fail'>DB Connection: FAILED</div>", unsafe_allow_html=True)
        # also show the error in a highlighted block for easier reading
        st.sidebar.markdown(f"<div class='db-err-block'>{str(err)}</div>", unsafe_allow_html=True)
    elif ok:
        st.sidebar.markdown("<div class='db-badge db-badge-success'>DB Connection: OK</div>", unsafe_allow_html=True)

    

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
                            # Show generated or database DDL for the selected table
                            try:
                                ddl = get_object_ddl(engine, selected)
                                if ddl:
                                    with st.expander("DDL / Create statement"):
                                        st.code(ddl, language="sql")
                            except Exception:
                                pass
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

                        # Download as CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download preview as CSV", data=csv, file_name=f"{selected.replace('.','_')}_preview.csv", mime='text/csv')

                    # --- Views: allow previewing views in the same database ---
                    views = list_views(engine)
                    st.subheader("Available views")
                    if not views:
                        st.write("(no views found or cannot connect)")
                    else:
                        selected_view = st.selectbox("Choose a view to preview", options=views)
                        if selected_view:
                            st.subheader(f"Preview: {selected_view}")

                            # Show schema and row count for view
                            v_schema_df = get_table_schema(engine, selected_view)
                            v_row_count = get_table_row_count(engine, selected_view)
                            vcol1, vcol2 = st.columns([3,1])
                            with vcol1:
                                if not v_schema_df.empty:
                                    st.markdown("**Columns (name : type)**")
                                    st.write(v_schema_df.set_index('COLUMN_NAME')['DATA_TYPE'].to_dict())
                                    # Show DDL/definition for the selected view if available
                                    try:
                                        v_ddl = get_object_ddl(engine, selected_view)
                                        if v_ddl:
                                            with st.expander("DDL / Definition"):
                                                st.code(v_ddl, language="sql")
                                    except Exception:
                                        pass
                                else:
                                    st.info("Could not retrieve column schema for view")
                            with vcol2:
                                st.markdown("**Row count**")
                                st.write(v_row_count if v_row_count is not None else "unknown")

                            st.markdown("---")
                            st.markdown("**Preview options**")
                            v_max_rows = st.number_input("Limit view rows", min_value=10, max_value=5000, value=50, step=10, key="view_limit")

                            v_df = read_table(engine, selected_view, limit=int(v_max_rows), where=None)
                            if v_df is None or v_df.empty:
                                st.info("No rows available for this view or failed to read.")
                            else:
                                st.dataframe(v_df)

                                # Download as CSV for view
                                v_csv = v_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Download view preview as CSV", data=v_csv, file_name=f"{selected_view.replace('.','_')}_preview.csv", mime='text/csv')

    elif page == "Dashboards":
        st.header("Embedded Dashboard")
        st.markdown("This page displays the Tableau dashboard.")
        # Try rendering the raw embed snippet first. Some Tableau embeds require
        # external JS that may take longer to load or be blocked by browser
        # extensions/CSP. Use a taller height and allow scrolling while
        # troubleshooting; once working you can reduce height/disable scrolling.
        try:
            # Use the render_tableau_embed helper so the embed is made responsive
            render_tableau_embed(TABLEAU_EMBED_HTML, height=1400)

            # Provide a troubleshooting expander so the user can inspect the
            # raw snippet and try a fallback iframe render if needed.
            with st.expander("Troubleshoot embed (show raw snippet / try fallback)"):
                st.markdown("If the dashboard does not appear, copy the raw embed HTML and check the browser console for errors (CSP/script load issues).")
                st.code(TABLEAU_EMBED_HTML[:1000] + ("..." if len(TABLEAU_EMBED_HTML) > 1000 else ""), language="html")
                if st.button("Render fallback iframe", key="render_tableau_fallback"):
                    # Try a safer iframe-only render in case the full embed HTML
                    # with script tags is being blocked. Attempt to extract a
                    # plausible URL by searching for 'src=' or host_url param.
                    def extract_url_from_embed(embed_html: str) -> str | None:
                        # look for a src="..." inside the snippet
                        import re
                        m = re.search(r"src=\"([^\"]+)\"", embed_html)
                        if m:
                            return m.group(1)
                        # try to extract host_url param value (url encoded)
                        m2 = re.search(r"<param name='host_url' value='([^']+)'", embed_html)
                        if m2:
                            return m2.group(1)
                        return None

                    url = extract_url_from_embed(TABLEAU_EMBED_HTML)
                    if url:
                        st.info("Attempting iframe fallback render (may require a public URL).")
                        # If we found a URL, render it in an iframe via helper
                        try:
                            render_tableau_embed(url)
                        except Exception as e:
                            st.error(f"Fallback iframe render failed: {e}")
                    else:
                        st.warning("Could not find a fallback URL inside the embed snippet. Check browser console for script errors.")

        except Exception as e:
            st.error(f"Failed to render embedded dashboard: {e}")

    elif page == "Gallery":
        st.header("Gallery — Parks")
        st.markdown("Images with title (full name, state) — sourced from the `park` table.")

        if engine is None:
            st.warning("Database not connected. Provide credentials in environment or .env.")
        else:
            # Simplified: only use the park table for gallery content. Prefer fully-qualified 'dbo.park',
            # then a plain 'park' table, or any table that ends with '.park'. Do NOT scan arbitrary tables.
            tables = list_tables(engine)
            chosen_table = None
            if tables:
                lower = [t.lower() for t in tables]
                if "dbo.park" in lower:
                    chosen_table = tables[lower.index("dbo.park")]
                elif "park" in lower:
                    chosen_table = tables[lower.index("park")]
                else:
                    # last resort: any table that ends with '.park'
                    for t in tables:
                        if t.lower().endswith('.park'):
                            chosen_table = t
                            break

            if not chosen_table:
                st.info("No `park` table found. Gallery requires a table named `park` or `dbo.park`.")
            else:
                limit = st.number_input("Limit items", min_value=6, max_value=1000, value=60, step=6, key="gallery_limit")
                # Read a lightweight preview (only rows) and then resolve simple columns from returned DataFrame
                preview_df = read_table(engine, chosen_table, limit=int(limit), where=None)
                if preview_df is None or preview_df.empty:
                    st.info("No rows available in the park table or failed to read.")
                else:
                    available_cols = preview_df.columns.tolist()

                    # Resolve common column names (simple, case-insensitive)
                    def resolve(col_candidates):
                        for cand in col_candidates:
                            for a in available_cols:
                                if a.lower() == cand.lower():
                                    return a
                        return None

                    label_actual = resolve(["fullname", "full_name", "name", "park_name"]) or (available_cols[0] if available_cols else None)
                    state_actual = resolve(["state", "st", "region"]) or None
                    img_actual = resolve(["images", "image", "img", "photo", "picture", "thumbnail", "thumb", "url", "logo"]) or None

                    st.markdown(f"**Gallery source:** `{chosen_table}` — showing up to {limit} items")

                    records = []
                    for _, r in preview_df.iterrows():
                        title_parts = []
                        if label_actual and pd.notna(r.get(label_actual)):
                            title_parts.append(str(r.get(label_actual)))
                        if state_actual and pd.notna(r.get(state_actual)):
                            title_parts.append(str(r.get(state_actual)))
                        title = ", ".join(title_parts) if title_parts else ""
                        img_val = r.get(img_actual) if img_actual in preview_df.columns else None
                        if img_val is None or (isinstance(img_val, float) and pd.isna(img_val)):
                            continue
                        records.append({"title": title, "img": img_val})

                    if not records:
                        st.info("No image data found in the park table.")
                    else:
                        per_row = 3
                        num = len(records)
                        for i in range(0, num, per_row):
                            chunk = records[i:i+per_row]
                            cols_iter = st.columns(len(chunk))
                            for cslot, item in zip(cols_iter, chunk):
                                with cslot:
                                    label = item.get("title", "")
                                    img_val = item.get("img")
                                    if isinstance(img_val, (bytes, bytearray, memoryview)):
                                        try:
                                            st.image(io.BytesIO(img_val), caption=label, use_column_width=True)
                                        except Exception:
                                            st.write(label)
                                            st.write("(unable to render binary image)")
                                    elif isinstance(img_val, str):
                                        v = img_val.strip()
                                        if v.startswith("data:") or v.startswith("http://") or v.startswith("https://"):
                                            try:
                                                st.image(v, caption=label, use_column_width=True)
                                            except Exception:
                                                st.write(label)
                                                st.write("(broken image URL)")
                                        else:
                                            try:
                                                b = base64.b64decode(v)
                                                st.image(io.BytesIO(b), caption=label, use_column_width=True)
                                            except Exception:
                                                st.write(label)
                                                st.code(v[:200] + ("..." if len(v) > 200 else ""))
                                    else:
                                        st.write(label)
                                        st.write("(unsupported image type)")

    else:
        st.header("Ask — Analytical Questions")
        st.markdown(
            "This page lists 8 analytical questions with SQL snippet hints. "
            "Click an item to view the SQL and press `Run SQL` to execute it against the connected database. "
        )

        max_rows = st.number_input("Max rows to display", min_value=10, max_value=2000, value=200)

        if engine is None:
            st.warning("Database not connected. Provide credentials in environment or .env.")

        # Load analytical questions from a JSON file if present, else try questions.py
        questions = []
        try:
            import json
            qpath = Path(__file__).parent / "questions.json"
            if qpath.exists():
                with open(qpath, "r", encoding="utf-8") as _f:
                    loaded = json.load(_f)
                    if isinstance(loaded, list):
                        questions = loaded
                    else:
                        st.warning("questions.json found but does not contain a list. No prebuilt questions loaded.")
            else:
                # Fallback: try legacy Python module if JSON not present
                try:
                    from questions import QUESTIONS as _q
                    if isinstance(_q, list):
                        questions = _q
                except Exception:
                    # no JSON and no questions.py — leave questions empty and show a notice below when rendering
                    questions = []
        except Exception as _e:
            # If anything went wrong loading/parsing, fall back to empty list and show a short warning
            try:
                st.warning(f"Failed to load questions.json: {_e}")
            except Exception:
                pass
            questions = []

        # Index hint removed per user request

        for i, q in enumerate(questions):
            with st.expander(f"{i+1}. {q['title']}"):
                # Provide an editable SQL area pre-populated with the snippet
                sql_key = f"sql_{q['id']}"
                # Use existing session_state value if present so reset callback can update it before render
                initial_sql = st.session_state.get(sql_key, q["sql"])
                sql_text = st.text_area("SQL (editable)", value=initial_sql, key=sql_key, height=220)

                # Offer a small reset button to restore the original snippet
                reset_col, run_col = st.columns([1, 1])
                with reset_col:
                    st.button("Reset to snippet", key=f"reset_{q['id']}", on_click=_reset_sql_in_session, args=(sql_key, q["sql"]))
                with run_col:
                    if st.button("Run SQL", key=f"run_{q['id']}"):
                        if engine is None:
                            st.error("No DB connection available. Check DB credentials.")
                        else:
                            try:
                                # Sanitize and possibly limit the SQL to avoid large scans
                                try:
                                    safe_sql = sanitize_sql_for_run(sql_text, max_limit=500)
                                except Exception as se:
                                    st.error(f"SQL validation failed: {se}")
                                    safe_sql = None

                                if not safe_sql:
                                    # validation failed; skip execution
                                    continue

                                # Execute the sanitized SQL
                                df = run_sql(engine, safe_sql)

                                if df is None or df.empty:
                                    st.info("Query returned no rows.")
                                else:
                                    left_col, mid_col, right_col = st.columns([1, 1, 1])
                                    with left_col:
                                        st.caption("Executed SQL")
                                        # show the actual SQL executed (may differ from editable text if sanitized)
                                        st.code(safe_sql, language="sql")

                                    with mid_col:
                                        st.caption("Table preview")
                                        if df.shape[0] > int(max_rows):
                                            st.warning(f"Query returned {df.shape[0]} rows; showing first {max_rows}.")
                                            st.dataframe(df.head(int(max_rows)), use_container_width=True)
                                        else:
                                            st.dataframe(df, use_container_width=True)

                                    with right_col:
                                        st.caption("Sample visualization")
                                        try:
                                            render_sample_viz(q['id'], df)
                                        except Exception as viz_e:
                                            st.write(f"Visualization failed: {viz_e}")
                            except Exception as e:
                                st.error(f"Failed to execute query: {e}")

                st.write("Use the SQL snippet above as a hint. Edit the SQL as needed before running on your database.")

        # Index hint display removed.


if __name__ == "__main__":
    main()
      
      