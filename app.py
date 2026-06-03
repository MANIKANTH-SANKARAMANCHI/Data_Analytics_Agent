"""
AI Data Analyst Agent - Professional Analytics Dashboard
Modular Streamlit app with sidebar navigation and structured panels.
"""

import html
import io
import json
import streamlit as st
import pandas as pd

from data_cleaning import clean_data
from data_quality import build_data_quality_insights
from eda import generate_summary, describe_data, compute_correlation
from visualization import (
    plot_histogram,
    plot_box,
    plot_distribution,
    plot_bar_categorical,
    plot_frequency_categorical,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_feature_importance,
)

try:
    from visualization import build_report_chart_pngs
except ImportError:
    # Older `visualization.py` without PDF chart helpers, or stale bytecode.
    def build_report_chart_pngs(_df, _feature_importances=None):
        return []
from model_training import generate_simple_ml_insights, train_models
from ai_insights import generate_ai_insights
from utils import (
    get_categorical_columns,
    get_id_columns,
    get_numeric_columns,
    resolve_target_in_cleaned,
)


# ---------------------------------------------------------------------------
# Page config and custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataPilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --primary:#6C5CE7;
  --bg:#F3F4F6;
  --card:rgba(255,255,255,0.92);
  --text:#111827;
  --muted:#6B7280;
  --border:rgba(17,24,39,0.08);
  --shadow:0 10px 30px rgba(17,24,39,0.08);
  --radius:12px;
}

html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; }
body { background: var(--bg); }

/* Center + constrain main content */
div.block-container{
  max-width: 1180px;
  padding-top: 0.25rem;
  padding-bottom: 2.5rem;
}

/* Hide Streamlit default top space a bit */
header[data-testid="stHeader"]{ background: transparent; }

/* Sidebar: fixed dark gradient */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #221D4F 0%, #2C2B8F 45%, #1B4BB3 100%);
  border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] *{
  color: rgba(255,255,255,0.92);
}
section[data-testid="stSidebar"] a{ color: rgba(255,255,255,0.92) !important; }

/* Force visible placeholder/value text in sidebar selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] *{
  color: #111827 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Sidebar controls */
div[data-testid="stSidebar"] .stButton>button{
  width:100%;
  border-radius: 10px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.18);
  color: rgba(255,255,255,0.92);
  padding: .65rem .9rem;
}
div[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background: linear-gradient(90deg, rgba(108,92,231,1) 0%, rgba(76,110,245,1) 100%);
  border: 0;
}

/* Card wrapper */
.saas-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.05rem 1.15rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(10px);
}
.card-title{
  font-size: .95rem;
  font-weight: 650;
  color: var(--text);
  margin: 0 0 .75rem 0;
}
.card-subtle{
  color: var(--muted);
  font-size: .85rem;
  margin-top: -.35rem;
}

/* Top navbar */
.topbar{
  position: sticky;
  top: 0.75rem;
  z-index: 50;
  display:flex;
  gap: 1rem;
  align-items:center;
  justify-content: space-between;
  padding: 0.9rem 1.1rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.75);
  backdrop-filter: blur(14px);
  box-shadow: 0 6px 24px rgba(17,24,39,0.08);
  margin-bottom: 1rem;
}
.topbar-title{
  font-size: 1.1rem;
  font-weight: 750;
  color: var(--text);
  letter-spacing: -0.01em;
}
.topbar-right{
  display:flex;
  gap: .75rem;
  align-items:center;
}
.search-pill{
  width: 360px;
  max-width: 42vw;
  border-radius: 999px;
  border: 1px solid var(--border);
  padding: .55rem .9rem;
  background: rgba(255,255,255,0.9);
  color: var(--muted);
  font-size: .9rem;
}
.profile-pill{
  display:flex;
  align-items:center;
  gap: .6rem;
  padding: .45rem .65rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.88);
}
.avatar{
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: linear-gradient(135deg, #6C5CE7 0%, #00D2FF 100%);
}
.profile-meta{
  line-height: 1.05;
}
.profile-name{ font-weight: 650; font-size: .85rem; color: var(--text); }
.profile-role{ font-size: .72rem; color: var(--muted); }

/* KPI bento strip — asymmetric grid */
.kpi-bento-wrap{
  margin-bottom: 1.25rem;
  padding: 1rem 1.05rem 1.05rem;
  border-radius: 20px;
  background:
    radial-gradient(ellipse 120% 80% at 0% 0%, rgba(108,92,231,0.14) 0%, transparent 55%),
    radial-gradient(ellipse 90% 70% at 100% 100%, rgba(0,184,148,0.10) 0%, transparent 50%),
    linear-gradient(180deg, rgba(255,255,255,0.65) 0%, rgba(243,244,246,0.9) 100%);
  border: 1px solid rgba(17,24,39,0.07);
  box-shadow: 0 16px 40px rgba(17,24,39,0.06);
}
.kpi-bento{
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr);
  grid-template-rows: auto auto;
  gap: 14px;
  align-items: stretch;
}
.kpi-cell{
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-cell:hover{
  transform: translateY(-3px);
  box-shadow: 0 14px 36px rgba(17,24,39,0.12);
}
.kpi-watermark{
  position: absolute;
  right: -0.15rem;
  bottom: -0.35rem;
  font-size: 4.25rem;
  line-height: 1;
  opacity: 0.11;
  pointer-events: none;
  user-select: none;
}
.kpi-cell-hero{
  grid-column: 1;
  grid-row: 1 / span 2;
  padding: 1.15rem 1.2rem 1.25rem;
  background: linear-gradient(155deg, #3d2f7a 0%, #5b4bc4 42%, #7c6ae8 100%);
  color: #fff;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 12px 32px rgba(76,63,145,0.35);
}
.kpi-cell-hero .kpi-hero-meta{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.65rem;
}
.kpi-pill{
  font-size: 0.68rem;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.28rem 0.55rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.2);
}
.kpi-live{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.72rem;
  font-weight: 600;
  opacity: 0.92;
}
.kpi-live-dot{
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: rgba(255,255,255,0.35);
  box-shadow: 0 0 0 3px rgba(255,255,255,0.12);
}
.kpi-live-dot.on{
  background: #55efc4;
  box-shadow: 0 0 0 3px rgba(85,239,196,0.35);
  animation: kpiPulse 2s ease-in-out infinite;
}
@keyframes kpiPulse{
  0%, 100%{ opacity: 1; transform: scale(1); }
  50%{ opacity: 0.85; transform: scale(0.92); }
}
.kpi-hero-value{
  font-size: clamp(2.1rem, 4vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
}
.kpi-hero-label{
  margin-top: 0.35rem;
  font-size: 0.88rem;
  font-weight: 500;
  color: rgba(255,255,255,0.78);
  max-width: 11rem;
  line-height: 1.35;
}
.kpi-cell-tile{
  padding: 0.95rem 1rem;
  background: rgba(255,255,255,0.94);
  border: 1px solid rgba(17,24,39,0.08);
  box-shadow: 0 8px 22px rgba(17,24,39,0.06);
}
.kpi-cell-tile .kpi-tile-top{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
}
.kpi-tile-icon{
  font-size: 1.35rem;
  line-height: 1;
  opacity: 0.92;
}
.kpi-tile-value{
  font-size: 1.65rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.kpi-tile-label{
  margin-top: 0.35rem;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--muted);
  line-height: 1.3;
}
.kpi-tile-accent{
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  border-radius: 16px 0 0 16px;
}
.kpi-tile-teal .kpi-tile-accent{ background: linear-gradient(180deg, #00b894, #55efc4); }
.kpi-tile-blue .kpi-tile-accent{ background: linear-gradient(180deg, #0984e3, #74b9ff); }
.kpi-cell-wide{
  grid-column: 2 / span 2;
  grid-row: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.9rem 1.1rem;
  background: linear-gradient(105deg, rgba(214,48,49,0.07) 0%, rgba(255,255,255,0.95) 38%);
  border: 1px solid rgba(214,48,49,0.18);
  box-shadow: 0 8px 22px rgba(214,48,49,0.08);
}
.kpi-wide-left{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
}
.kpi-wide-icon{
  font-size: 1.6rem;
  line-height: 1;
}
.kpi-wide-copy .kpi-wide-value{
  font-size: 1.55rem;
  font-weight: 800;
  color: #c0392b;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.kpi-wide-copy .kpi-wide-label{
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--muted);
  margin-top: 0.15rem;
}
.kpi-wide-hint{
  font-size: 0.72rem;
  color: var(--muted);
  max-width: 14rem;
  text-align: right;
  line-height: 1.35;
}
@media (max-width: 720px){
  .kpi-bento{
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto;
  }
  .kpi-cell-hero{
    grid-column: 1 / span 2;
    grid-row: 1;
  }
  .kpi-cell-tile.kpi-insights-slot{ grid-column: 1; grid-row: 2; }
  .kpi-cell-tile.kpi-models-slot{ grid-column: 2; grid-row: 2; }
  .kpi-cell-wide{
    grid-column: 1 / span 2;
    grid-row: 3;
    flex-direction: column;
    align-items: flex-start;
  }
  .kpi-wide-hint{ text-align: left; max-width: none; }
}

/* Make file uploader look like drag/drop box */
div[data-testid="stFileUploader"]{
  border: 1.5px dashed rgba(108,92,231,0.55);
  background: rgba(108,92,231,0.06);
  border-radius: var(--radius);
  padding: 1.1rem 1.1rem;
}
div[data-testid="stFileUploader"] label{ display:none; }
div[data-testid="stFileUploader"] button{
  border-radius: 10px !important;
}

/* Tighten radio items in sidebar */
div[data-testid="stSidebar"] [role="radiogroup"] label{
  padding: .35rem .35rem;
  border-radius: 10px;
}

/* Expander polish */
div[data-testid="stExpander"]{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 6px 16px rgba(17,24,39,0.06);
  background: rgba(255,255,255,0.86);
}

/* Export Results — card row (Tailwind-like: rounded-2xl, glass, hover lift) */
.export-grid-spacer{ height: 0.25rem; }
.export-card-head{
  background: rgba(255,255,255,0.70);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(17,24,39,0.08);
  border-bottom: none;
  border-radius: 1rem 1rem 0 0;
  padding: 1.35rem 1.35rem 1.1rem;
  box-shadow: 0 6px 24px rgba(17,24,39,0.07);
  transition: box-shadow 0.22s ease, transform 0.22s ease;
}
.export-card-top{
  display: flex;
  gap: 0.95rem;
  align-items: flex-start;
}
.export-card-icon{
  font-size: 1.85rem;
  line-height: 1;
  flex-shrink: 0;
  filter: drop-shadow(0 2px 6px rgba(17,24,39,0.08));
}
.export-card-title{
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
  margin: 0 0 0.4rem 0;
  line-height: 1.2;
}
.export-card-desc{
  font-size: 0.82rem;
  font-weight: 450;
  color: var(--muted);
  line-height: 1.45;
  margin: 0;
}
.export-card-status{
  margin-top: 1.05rem;
  padding-top: 0.9rem;
  border-top: 1px solid rgba(17,24,39,0.07);
}
.export-status{
  display: inline-flex;
  align-items: center;
  font-size: 0.78rem;
  font-weight: 650;
  padding: 0.38rem 0.72rem;
  border-radius: 999px;
  letter-spacing: 0.01em;
}
.export-status--ok{
  background: rgba(0, 184, 148, 0.14);
  color: #0d7a67;
}
.export-status--warn{
  background: rgba(214, 48, 49, 0.11);
  color: #b83232;
}
.export-status--muted{
  background: rgba(107, 114, 128, 0.13);
  color: var(--muted);
}
.block-container div.stColumn:has(.export-card-head){
  transition: transform 0.22s ease;
}
.block-container div.stColumn:has(.export-card-head):hover{
  transform: translateY(-4px);
}
.block-container div.stColumn:has(.export-card-head):hover .export-card-head{
  box-shadow: 0 16px 40px rgba(17,24,39,0.12);
}
.block-container div.stColumn:has(.export-card-head) .stDownloadButton > button{
  width: 100% !important;
  border-radius: 0 0 1rem 1rem !important;
  border-top: 1px solid rgba(17,24,39,0.06) !important;
  box-shadow: 0 8px 26px rgba(17,24,39,0.08) !important;
  padding-top: 0.65rem !important;
  padding-bottom: 0.65rem !important;
  font-weight: 600 !important;
  transition: box-shadow 0.22s ease, transform 0.22s ease, filter 0.22s ease !important;
}
.block-container div.stColumn:has(.export-card-head) .stButton > button{
  width: 100% !important;
  border-radius: 0 0 1rem 1rem !important;
  border-top: 1px solid rgba(17,24,39,0.06) !important;
  box-shadow: 0 8px 26px rgba(17,24,39,0.06) !important;
  padding-top: 0.65rem !important;
  padding-bottom: 0.65rem !important;
  font-weight: 600 !important;
  transition: box-shadow 0.22s ease, transform 0.22s ease !important;
}
.block-container div.stColumn:has(.export-card-head):hover .stDownloadButton > button,
.block-container div.stColumn:has(.export-card-head):hover .stButton > button{
  box-shadow: 0 14px 38px rgba(17,24,39,0.14) !important;
}
@media (max-width: 900px){
  .block-container div.stColumn:has(.export-card-head){ margin-bottom: 0.75rem; }
}
</style>
    """,
    unsafe_allow_html=True,
)


def _card_open(title: str | None = None, subtitle: str | None = None):
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-subtle">{subtitle}</div>', unsafe_allow_html=True)


def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def _json_numpy_serial(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(type(obj).__name__)


def _export_card_head(
    icon: str,
    title: str,
    description: str,
    *,
    status_kind: str,
    status_text: str,
) -> None:
    """Premium export card header (icon, title, description, status pill)."""
    status_class = {
        "ok": "export-status--ok",
        "warn": "export-status--warn",
        "muted": "export-status--muted",
    }.get(status_kind, "export-status--muted")
    ti = html.escape(title)
    de = html.escape(description)
    status_esc = html.escape(status_text)
    st.markdown(
        f"""<div class="export-card-head">
  <div class="export-card-top">
    <span class="export-card-icon" aria-hidden="true">{icon}</span>
    <div>
      <div class="export-card-title">{ti}</div>
      <p class="export-card-desc">{de}</p>
    </div>
  </div>
  <div class="export-card-status"><span class="export-status {status_class}">{status_esc}</span></div>
</div>""",
        unsafe_allow_html=True,
    )


def _render_kpi_bento(
    datasets: int,
    insights: int,
    models: int,
    anomalies: int,
) -> str:
    """Asymmetric bento layout for workspace KPIs (single HTML block)."""
    ds = html.escape(str(datasets))
    ins = html.escape(str(insights))
    mod = html.escape(str(models))
    ano = html.escape(str(anomalies))
    live_class = " on" if datasets else ""
    live_text = "Dataset loaded" if datasets else "Waiting for upload"

    return f"""
<div class="kpi-bento-wrap">
  <div class="kpi-bento">
    <div class="kpi-cell kpi-cell-hero">
      <span class="kpi-watermark" aria-hidden="true">🗂️</span>
      <div class="kpi-hero-meta">
        <span class="kpi-pill">Workspace</span>
        <span class="kpi-live"><span class="kpi-live-dot{live_class}"></span>{html.escape(live_text)}</span>
      </div>
      <div class="kpi-hero-value">{ds}</div>
      <div class="kpi-hero-label">Datasets processed — your files in this session</div>
    </div>
    <div class="kpi-cell kpi-cell-tile kpi-tile-teal kpi-insights-slot">
      <span class="kpi-tile-accent" aria-hidden="true"></span>
      <span class="kpi-watermark" aria-hidden="true">✨</span>
      <div class="kpi-tile-top">
        <div>
          <div class="kpi-tile-value">{ins}</div>
          <div class="kpi-tile-label">Insights generated</div>
        </div>
        <span class="kpi-tile-icon">✨</span>
      </div>
    </div>
    <div class="kpi-cell kpi-cell-tile kpi-tile-blue kpi-models-slot">
      <span class="kpi-tile-accent" aria-hidden="true"></span>
      <span class="kpi-watermark" aria-hidden="true">🧩</span>
      <div class="kpi-tile-top">
        <div>
          <div class="kpi-tile-value">{mod}</div>
          <div class="kpi-tile-label">Models built</div>
        </div>
        <span class="kpi-tile-icon">🧩</span>
      </div>
    </div>
    <div class="kpi-cell kpi-cell-wide">
      <div class="kpi-wide-left">
        <span class="kpi-wide-icon" aria-hidden="true">🚨</span>
        <div class="kpi-wide-copy">
          <div class="kpi-wide-value">{ano}</div>
          <div class="kpi-wide-label">Anomalies detected</div>
        </div>
      </div>
      <div class="kpi-wide-hint">Cleaning pipeline: duplicates removed + missing cells flagged before training.</div>
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state():
    keys_defaults = {
        "uploaded_df": None,
        "cleaned_df": None,
        "cleaning_report": None,
        "summary": None,
        "desc": None,
        "corr": None,
        "model_results": None,
        "feature_importances": None,
        "problem_type": None,
        "target_column": None,
        "analysis_done": False,
        "ai_insights_text": None,
        "ml_quick_insights": None,
        "ml_training_note": None,
        "ml_pipeline_status": None,
    }
    for key, default in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Core pipeline functions (modular)
# ---------------------------------------------------------------------------

def load_data(uploaded_file):
    """Load CSV or Excel from uploaded file into a DataFrame."""
    if uploaded_file is None:
        return None

    allowed_mime_types = {
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    file_name = (uploaded_file.name or "").lower()
    file_mime = (uploaded_file.type or "").lower()
    is_csv = file_name.endswith(".csv")
    is_xlsx = file_name.endswith(".xlsx")

    # Validate type before parsing so users only see errors for unsupported files.
    if not (is_csv or is_xlsx):
        st.error("Unsupported file type. Please upload a CSV (.csv) or Excel (.xlsx) file.")
        return None
    if file_mime and file_mime not in allowed_mime_types:
        st.error("Unsupported file type. Please upload a CSV (.csv) or Excel (.xlsx) file.")
        return None

    try:
        if is_csv:
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except ImportError:
        st.error(
            "Excel support requires the 'openpyxl' package. Install it and restart the app."
        )
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def clean_data_pipeline(df: pd.DataFrame):
    """Run data cleaning and return cleaned DataFrame and report."""
    return clean_data(df)


def run_eda(df: pd.DataFrame):
    """Run exploratory data analysis: summary, describe, correlation."""
    summary = generate_summary(df)
    desc = describe_data(df)
    corr = compute_correlation(df)
    return summary, desc, corr


def train_models_pipeline(df: pd.DataFrame, target_column: str | None):
    """Train models based on detected problem type (target must exist on df columns)."""
    if target_column is None or target_column not in df.columns:
        return None, None, None, None
    result = train_models(df, target_column)
    # Always return a 4-tuple (handles stale reloads if an older train_models returned 3 values).
    if isinstance(result, tuple) and len(result) == 4:
        return result
    if isinstance(result, tuple) and len(result) == 3:
        a, b, c = result
        return a, b, c, None
    return None, None, None, None


def can_train_models(df: pd.DataFrame, target_column: str | None) -> tuple[bool, str]:
    """Check whether the dataset has usable feature columns for training."""
    if df is None or df.empty:
        return False, "No dataset available for training."
    if target_column is None or target_column not in df.columns:
        return False, "Select a valid target column to train models."

    id_columns = get_id_columns(df)
    drop_cols = list({target_column} | {c for c in id_columns if c in df.columns})
    X = df.drop(columns=drop_cols)
    if X.shape[1] == 0:
        return (
            False,
            "No usable feature columns remain after excluding target/ID columns. "
            "Upload a dataset with additional predictor columns.",
        )

    X_encoded = pd.get_dummies(X, drop_first=True)
    if X_encoded.shape[1] == 0:
        return (
            False,
            "Feature encoding produced no usable columns. Choose a different target or dataset.",
        )

    return True, ""


def generate_ai_insights_pipeline():
    """Generate LLM insights from summary, stats, correlation, visualizations, and model results."""
    summary = st.session_state.get("summary")
    desc = st.session_state.get("desc")
    corr = st.session_state.get("corr")
    model_results = st.session_state.get("model_results")
    problem_type = st.session_state.get("problem_type")
    cleaned_df = st.session_state.get("cleaned_df")
    feature_importances = st.session_state.get("feature_importances")
    if summary is None or desc is None:
        return None

    # Build a lightweight textual summary of the visualizations that the app produces
    numeric_cols = summary.get("numeric_columns", []) or []
    categorical_cols = summary.get("categorical_columns", []) or []

    viz_lines: list[str] = []
    if cleaned_df is not None:
        viz_lines.append(
            f"Histograms, boxplots, and distribution plots are available for numeric columns: "
            f"{', '.join(numeric_cols) if numeric_cols else 'no numeric columns detected'}."
        )
        viz_lines.append(
            f"Bar and frequency charts are available for categorical columns: "
            f"{', '.join(categorical_cols) if categorical_cols else 'no categorical columns detected'}."
        )
    else:
        viz_lines.append("No cleaned DataFrame available to generate visualization summaries.")

    if corr is not None and not corr.empty:
        # Identify a few of the strongest correlation pairs for narrative context
        corr_matrix = corr.copy()
        corr_pairs = []
        cols = list(corr_matrix.columns)
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if j <= i:
                    continue
                val = corr_matrix.loc[col_i, col_j]
                if pd.notnull(val):
                    corr_pairs.append((col_i, col_j, float(val)))
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        top_pairs = corr_pairs_sorted[:5]
        if top_pairs:
            desc_pairs = [
                f"{a} vs {b}: correlation {v:.3f}" for a, b, v in top_pairs
            ]
            viz_lines.append(
                "The correlation heatmap highlights the strongest relationships, including: "
                + "; ".join(desc_pairs)
                + "."
            )

    if feature_importances is not None and not feature_importances.empty:
        top_features = (
            feature_importances.sort_values("importance", ascending=False)
            .head(5)["feature"]
            .astype(str)
            .tolist()
        )
        if top_features:
            viz_lines.append(
                "A feature-importance bar chart shows the most influential features: "
                + ", ".join(top_features)
                + "."
            )

    visualization_summaries = " ".join(viz_lines).strip()

    try:
        return generate_ai_insights(
            summary=summary,
            stats_description=desc,
            correlation=corr,
            visualization_summaries=visualization_summaries,
            model_results=model_results,
            problem_type=problem_type,
        )
    except Exception as e:
        st.error(
            "Failed to generate AI insights. Set GEMINI_API_KEY in your environment."
        )
        st.exception(e)
        return None


def build_ai_insights_pdf_bytes(
    insights_text: str,
    chart_images: list[tuple[str, bytes]] | None = None,
) -> bytes | None:
    """Render LLM AI insights narrative into a multi-page UTF-8 PDF, optional chart PNGs."""
    text = (insights_text or "").strip()
    if not text:
        return None

    from pathlib import Path

    import matplotlib.font_manager as fm
    from fpdf import FPDF

    regular = Path(
        fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    )
    bold = Path(
        fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    )
    if not regular.is_file() or not bold.is_file():
        return None

    class Doc(FPDF):
        def header(self) -> None:
            self.set_font("DejaVuSans", "B", 14)
            self.cell(
                0,
                10,
                "Full AI Analytics Report",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )
            self.set_font("DejaVuSans", size=9)
            self.cell(
                0,
                6,
                "Insights synthesized from the dataset, EDA, visualizations, and ML results",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )
            self.ln(6)

    pdf = Doc()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_font("DejaVuSans", "", str(regular))
    pdf.add_font("DejaVuSans", "B", str(bold))
    pdf.add_page()
    pdf.set_font("DejaVuSans", size=11)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pdf.multi_cell(0, 6, normalized)

    if chart_images:
        pdf.ln(6)
        pdf.set_font("DejaVuSans", "B", 13)
        pdf.multi_cell(
            0,
            8,
            "Visual appendix — key charts from this analysis",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)
        img_w = max(40.0, pdf.epw - 30)
        for title, img_bytes in chart_images:
            if not img_bytes:
                continue
            pdf.set_font("DejaVuSans", "B", 11)
            pdf.multi_cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVuSans", size=9)
            try:
                pdf.image(io.BytesIO(img_bytes), x=15, w=img_w)
            except Exception:
                pdf.multi_cell(
                    0,
                    6,
                    "(Chart could not be embedded in this PDF.)",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            pdf.ln(6)

    raw = pdf.output()
    return bytes(raw) if raw else None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar(df_loaded: bool, analysis_done: bool):
    st.sidebar.markdown("## DataPilot")
    st.sidebar.caption("Premium analytics workspace")

    if st.sidebar.button("＋ New Project", type="primary", key="new_project_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.sidebar.markdown("---")

    menu = [
        ("Dashboard", "dashboard"),
        ("Data Insights", "insights"),
        ("Predictive Analysis", "predictive"),
        ("Data Quality", "anomalies"),
        ("AI Insights", "ai_insights"),
        ("Export Results", "reports"),
    ]
    key_to_label = {k: v for (v, k) in menu}

    choice = st.sidebar.radio(
        "Menu",
        options=[k for (_, k) in menu],
        format_func=lambda k: key_to_label[k],
        index=0,
        key="saas_nav_radio",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    if not df_loaded:
        st.sidebar.info("Upload a dataset to begin.")
    elif not analysis_done:
        st.sidebar.info("Run Full Analysis to unlock insights & models.")

    return choice


# ---------------------------------------------------------------------------
# Panel: Upload Dataset
# ---------------------------------------------------------------------------

def panel_upload():
    _card_open("Upload dataset", "Drag & drop a CSV/XLSX to start analysis.")
    uploaded_file = st.file_uploader(
        "Upload",
        type=["csv", "xlsx"],
        key="file_uploader",
        label_visibility="collapsed",
    )

    df = None
    if uploaded_file is not None:
        loaded_df = load_data(uploaded_file)
        if loaded_df is not None:
            st.session_state.uploaded_df = loaded_df
            df = loaded_df
    elif st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.info("Using previously uploaded dataset. Upload a new file to replace.")

    if df is not None:
        st.success(f"Dataset loaded: **{df.shape[0]}** rows × **{df.shape[1]}** columns")
        st.caption("Scrollable preview of the full dataset.")
        st.dataframe(df, use_container_width=True, height=420)
    _card_close()


# ---------------------------------------------------------------------------
# Panel: Data Overview
# ---------------------------------------------------------------------------

def panel_data_overview(df: pd.DataFrame):
    _card_open("Data overview", "Quick health check of the raw dataset.")

    if df is None or df.empty:
        st.warning("No dataset loaded.")
        _card_close()
        return

    # Metric row: shape, columns, missing, duplicates
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        missing_total = int(df.isnull().sum().sum())
        st.metric("Missing Values", missing_total)
    with c4:
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", int(dup_count))
    _card_close()

    # Column names and types in a dataframe
    _card_open("Schema", "Columns, dtypes, and missingness.")
    st.markdown("**Column names & data types**")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[col].dtype) for col in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("**Missing values per column**")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing count": [int(df[col].isnull().sum()) for col in df.columns],
    })
    missing_df = missing_df[missing_df["Missing count"] > 0]
    if missing_df.empty:
        st.caption("No missing values.")
    else:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    _card_close()


# ---------------------------------------------------------------------------
# Panel: Data Cleaning Report
# ---------------------------------------------------------------------------

def panel_cleaning_report():
    _card_open(
        "Data Quality & Cleaning Insights",
        "Duplicates, missing values, type fixes, normalization.",
    )

    report = st.session_state.get("cleaning_report")
    if report is None:
        st.info("Run Full Analysis to see the cleaning report.")
        _card_close()
        return

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dup = report.get("duplicates_removed", 0)
        st.markdown(
            f'<div class="metric-card"><h4>Removed duplicates</h4><p class="value">{dup}</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        before = report.get("missing_values_before", {})
        after = report.get("missing_values_after", {})
        total_before = sum(v for v in before.values() if isinstance(v, (int, float)))
        total_after = sum(v for v in after.values() if isinstance(v, (int, float)))
        filled = total_before - total_after
        st.markdown(
            f'<div class="metric-card"><h4>Handled missing values</h4><p class="value">{int(filled)} filled</p></div>',
            unsafe_allow_html=True,
        )

    with col3:
        conversions = report.get("type_conversions", [])
        converted = sum(1 for c in conversions if c.get("new_type") not in ("object (unchanged)",))
        st.markdown(
            f'<div class="metric-card"><h4>Converted categorical</h4><p class="value">{converted} columns</p></div>',
            unsafe_allow_html=True,
        )

    with col4:
        norm = report.get("normalized_columns", [])
        n_norm = len([n for n in norm if n.get("original") != n.get("normalized")]) or len(norm)
        st.metric("Normalized column names", int(n_norm))

    dq = build_data_quality_insights(report)

    st.subheader("Data Quality Score")
    st.metric("Score", f"{dq['score']}/100")

    st.subheader("Before vs After Cleaning")
    st.dataframe(dq["comparison_df"], use_container_width=True, hide_index=True)

    st.subheader("Issues Detected")
    if dq["issues"]:
        for issue in dq["issues"]:
            st.write(f"⚠️ {issue}")
    else:
        st.success("No major data quality issues detected")

    st.subheader("Impact on Machine Learning")
    for item in dq["impact"]:
        st.write(f"📊 {item}")

    st.subheader("Recommendations")
    for rec in dq["recommendations"]:
        st.write(f"💡 {rec}")

    _card_close()
    with st.expander("Full cleaning report (JSON)"):
        st.json(report)


# ---------------------------------------------------------------------------
# Panel: EDA
# ---------------------------------------------------------------------------

def panel_eda():
    _card_open("Exploratory data analysis", "Summary, descriptive stats, and correlation.")

    cleaned_df = st.session_state.get("cleaned_df")
    summary = st.session_state.get("summary")
    desc = st.session_state.get("desc")
    corr = st.session_state.get("corr")

    if cleaned_df is None or summary is None:
        st.info("Run Full Analysis to see EDA.")
        _card_close()
        return

    st.markdown("**Dataset summary**")
    summary_display = pd.DataFrame([
        {"Metric": "Rows", "Value": summary.get("rows", "")},
        {"Metric": "Columns", "Value": summary.get("columns", "")},
        {"Metric": "Numeric columns", "Value": ", ".join(summary.get("numeric_columns", [])) or "—"},
        {"Metric": "Categorical columns", "Value": ", ".join(summary.get("categorical_columns", [])) or "—"},
    ])
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    st.markdown("**Statistical description**")
    st.dataframe(desc, use_container_width=True)

    st.markdown("**Correlation matrix (numeric)**")
    if corr is not None and not corr.empty:
        st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
    else:
        st.caption("Not enough numeric columns for correlation.")
    _card_close()


# ---------------------------------------------------------------------------
# Panel: Visualizations (grid with Plotly)
# ---------------------------------------------------------------------------

def panel_visualizations():
    _card_open("Visualizations", "Interactive charts — use the tabs below to switch chart types.")

    cleaned_df = st.session_state.get("cleaned_df")
    if cleaned_df is None:
        st.info("Run Full Analysis to see visualizations.")
        _card_close()
        return

    numeric_cols = get_numeric_columns(cleaned_df)
    categorical_cols = get_categorical_columns(cleaned_df)

    v_corr, v_num, v_cat = st.tabs(["Correlation", "Numeric", "Categorical"])

    with v_corr:
        heatmap_fig = plot_correlation_heatmap(cleaned_df)
        if heatmap_fig is not None:
            st.plotly_chart(heatmap_fig, use_container_width=True, key="viz_heatmap")
        else:
            st.caption("Not enough numeric columns for a correlation heatmap.")

    with v_num:
        if numeric_cols:
            col_sel = st.selectbox(
                "Select numeric column", options=numeric_cols, key="viz_numeric_col"
            )
            graph_sel = st.selectbox(
                "Select graph type",
                options=["Histogram", "Box Plot", "Distribution"],
                key="viz_numeric_graph_sel",
            )
            if graph_sel == "Histogram":
                st.plotly_chart(
                    plot_histogram(cleaned_df, col_sel),
                    use_container_width=True,
                    key="viz_hist_sel",
                )
            elif graph_sel == "Box Plot":
                st.plotly_chart(
                    plot_box(cleaned_df, col_sel),
                    use_container_width=True,
                    key="viz_box_sel",
                )
            else:
                st.plotly_chart(
                    plot_distribution(cleaned_df, col_sel),
                    use_container_width=True,
                    key="viz_dist_sel",
                )
        else:
            st.caption("No numeric columns for charts.")

    with v_cat:
        if categorical_cols:
            cat_sel = st.selectbox(
                "Select categorical column", options=categorical_cols, key="viz_cat_col"
            )
            cat_graph_sel = st.selectbox(
                "Select graph type",
                options=["Bar Chart", "Frequency Chart"],
                key="viz_cat_graph_sel",
            )
            if cat_graph_sel == "Bar Chart":
                st.plotly_chart(
                    plot_bar_categorical(cleaned_df, cat_sel),
                    use_container_width=True,
                    key="viz_bar_cat",
                )
            else:
                st.plotly_chart(
                    plot_frequency_categorical(cleaned_df, cat_sel),
                    use_container_width=True,
                    key="viz_freq_cat",
                )
        else:
            st.caption("No categorical columns for charts.")

    _card_close()


# ---------------------------------------------------------------------------
# Panel: Machine Learning
# ---------------------------------------------------------------------------

def panel_ml():
    _card_open("Predictive analysis", "Train and compare models using your selected target.")

    cleaned_df = st.session_state.get("cleaned_df")
    if cleaned_df is None:
        st.info("Run Full Analysis with a target column selected to see models.")
        _card_close()
        return

    target_used = st.session_state.get("target_column")
    model_results = st.session_state.get("model_results")
    problem_type = st.session_state.get("problem_type")
    feature_importances = st.session_state.get("feature_importances")
    ml_quick = st.session_state.get("ml_quick_insights")

    st.subheader("Model results")
    st.write(f"**Target:** {target_used or '—'}")
    st.write(
        f"**Model type:** "
        f"{problem_type.replace('_', ' ').title() if problem_type else '—'}"
    )
    note = st.session_state.get("ml_training_note")
    if note:
        st.info(note)

    pipeline_status = st.session_state.get("ml_pipeline_status")
    if pipeline_status and not model_results:
        st.warning(pipeline_status)

    if model_results:
        st.markdown("**Metrics (hold-out test set)**")
        rows = []
        for model_name, metrics in model_results.items():
            if problem_type == "classification":
                rows.append({
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy", "—"),
                    "Precision": metrics.get("precision", "—"),
                    "Recall": metrics.get("recall", "—"),
                    "F1": metrics.get("f1_score", "—"),
                })
            else:
                rows.append({
                    "Model": model_name,
                    "R²": metrics.get("r2_score", "—"),
                    "RMSE": metrics.get("rmse", "—"),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if feature_importances is not None and not feature_importances.empty:
            st.markdown("**Feature importance** (Random Forest)")
            st.plotly_chart(
                plot_feature_importance(feature_importances),
                use_container_width=True,
                key="ml_feature_importance",
            )

        if ml_quick:
            st.markdown("**Quick insights**")
            for line in ml_quick:
                st.markdown(f"- {line}")

        with st.expander("Full model metrics (JSON)"):
            st.json(model_results)
    else:
        st.info("No model results yet. Run **Run Full Analysis** in the sidebar with a target selected.")
        if target_used is None:
            st.caption("Choose **Select Target Column** in the sidebar, then run the full pipeline.")
            _card_close()
            return

        cleaning_report = st.session_state.get("cleaning_report")
        target_in_cleaned = resolve_target_in_cleaned(
            cleaned_df, target_used, cleaning_report
        )
        can_train, train_msg = can_train_models(cleaned_df, target_in_cleaned)
        if not can_train:
            st.warning(train_msg)
            _card_close()
            return

        if st.button("Train Models Now", key="ml_train_now", type="primary"):
            with st.spinner("Training models..."):
                try:
                    (
                        model_results,
                        feature_importances,
                        problem_type,
                        training_note,
                    ) = train_models_pipeline(cleaned_df, target_in_cleaned)
                    st.session_state.model_results = model_results
                    st.session_state.feature_importances = feature_importances
                    st.session_state.problem_type = problem_type
                    st.session_state.ml_training_note = training_note
                    st.session_state.ml_pipeline_status = None
                    y_series = (
                        cleaned_df[target_in_cleaned]
                        if target_in_cleaned in cleaned_df.columns
                        else None
                    )
                    st.session_state.ml_quick_insights = generate_simple_ml_insights(
                        str(target_used),
                        problem_type or "",
                        model_results or {},
                        feature_importances,
                        y_series,
                    )
                    st.success("Model training complete.")
                    st.rerun()
                except Exception as e:
                    err_text = f"Model training failed: {e}"
                    st.error(err_text)
                    st.session_state.ml_pipeline_status = err_text
    _card_close()


# ---------------------------------------------------------------------------
# Panel: AI Insights
# ---------------------------------------------------------------------------

def panel_ai_insights():
    _card_open("AI insights", "Generate an executive-style narrative from the analysis.")

    if st.session_state.get("cleaned_df") is None:
        st.info("Run Full Analysis first.")
        _card_close()
        return

    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights..."):
            text = generate_ai_insights_pipeline()
            if text:
                st.session_state.ai_insights_text = text

    insights = st.session_state.get("ai_insights_text")
    if insights:
        st.markdown(
            "### AI Analytics Report\n\n"
            "Below is a structured report generated by the AI based on the dataset, EDA, visualizations, "
            "correlations, and machine learning results.",
        )
        st.markdown(insights)
    else:
        st.caption("Click 'Generate AI Insights' to get LLM-generated insights (requires GEMINI_API_KEY).")
    _card_close()


# ---------------------------------------------------------------------------
# Panel: Download Results
# ---------------------------------------------------------------------------

def panel_download():
    _card_open("Export Results")

    cleaned_df = st.session_state.get("cleaned_df")
    model_results = st.session_state.get("model_results")
    ai_insights = st.session_state.get("ai_insights_text")

    if cleaned_df is None:
        st.info("Run Full Analysis to enable downloads.")
        _card_close()
        return

    st.markdown('<div class="export-grid-spacer"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        _export_card_head(
            "📄",
            "Clean Dataset",
            "Processed and ready for analysis",
            status_kind="muted",
            status_text="Ready to export",
        )
        buf = io.StringIO()
        cleaned_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button(
            "Download CSV",
            data=buf.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="dl_csv",
            use_container_width=True,
            type="primary",
        )

    with col2:
        if model_results:
            _export_card_head(
                "🧩",
                "Model Results",
                "Metrics, predictions & feature importance",
                status_kind="ok",
                status_text="Results available",
            )
            st.download_button(
                "Download JSON",
                data=json.dumps(
                    model_results, indent=2, default=_json_numpy_serial
                ),
                file_name="model_results.json",
                mime="application/json",
                key="dl_models",
                use_container_width=True,
                type="primary",
            )
        else:
            _export_card_head(
                "🧩",
                "Model Results",
                "Metrics, predictions & feature importance",
                status_kind="warn",
                status_text="No model run yet",
            )
            if st.button(
                "Open Predictive Analysis",
                key="export_go_predictive",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.saas_nav_radio = "predictive"
                st.rerun()

    with col3:
        chart_pngs = build_report_chart_pngs(
            cleaned_df,
            st.session_state.get("feature_importances"),
        )
        pdf_bytes = (
            build_ai_insights_pdf_bytes(ai_insights, chart_pngs)
            if ai_insights
            else None
        )
        if pdf_bytes:
            _export_card_head(
                "✨",
                "AI Insights Report",
                "PDF includes the AI narrative plus embedded chart images (heatmap, sample plots, feature importance when available)",
                status_kind="ok",
                status_text="Report ready",
            )
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="full_ai_report.pdf",
                mime="application/pdf",
                key="dl_ai_pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            _export_card_head(
                "✨",
                "AI Insights Report",
                "PDF includes narrative text and a visual appendix with key charts after you generate insights",
                status_kind="warn",
                status_text="Insights not generated",
            )
            if st.button(
                "Generate Insights",
                key="export_go_ai_insights",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.saas_nav_radio = "ai_insights"
                st.rerun()

    _card_close()


# ---------------------------------------------------------------------------
# Main: Run Analysis button and routing
# ---------------------------------------------------------------------------

def main():
    init_session_state()

    df = st.session_state.uploaded_df
    df_loaded = df is not None and not df.empty
    analysis_done = st.session_state.analysis_done

    choice = render_sidebar(df_loaded, analysis_done)

    # KPI row (always visible)
    uploaded_flag = 1 if df_loaded else 0
    insights_flag = 1 if st.session_state.get("ai_insights_text") else 0
    model_results = st.session_state.get("model_results") or {}
    models_built = len(model_results) if isinstance(model_results, dict) else 0
    report = st.session_state.get("cleaning_report") or {}
    anomalies_detected = int(report.get("duplicates_removed", 0) or 0)
    try:
        before = report.get("missing_values_before", {}) or {}
        anomalies_detected += int(sum(v for v in before.values() if isinstance(v, (int, float))))
    except Exception:
        pass

    st.markdown(
        _render_kpi_bento(
            uploaded_flag,
            insights_flag,
            models_built,
            anomalies_detected,
        ),
        unsafe_allow_html=True,
    )

    # Run Full Analysis (only when dataset is loaded)
    if df_loaded:
        st.sidebar.markdown("**Analysis**")
        target_for_run = st.sidebar.selectbox(
            "Select Target Column (What do you want to predict?)",
            options=list(df.columns),
            index=None,
            placeholder="Choose an option",
            key="target_run",
        )

        if st.sidebar.button("Run Full Analysis", type="primary"):
            with st.spinner("Running pipeline: cleaning, EDA, and models..."):
                cleaned_df, cleaning_report = clean_data_pipeline(df)
                summary, desc, corr = run_eda(cleaned_df)

                target_in_cleaned = resolve_target_in_cleaned(
                    cleaned_df, target_for_run, cleaning_report
                )

                model_results = None
                feature_importances = None
                problem_type = None
                training_note = None
                ml_quick: list[str] | None = None
                ml_pipeline_status = None

                if target_for_run is not None and target_in_cleaned is not None:
                    can_train, train_msg = can_train_models(cleaned_df, target_in_cleaned)
                    if can_train:
                        try:
                            (
                                model_results,
                                feature_importances,
                                problem_type,
                                training_note,
                            ) = train_models_pipeline(cleaned_df, target_in_cleaned)
                            y_series = cleaned_df[target_in_cleaned]
                            ml_quick = generate_simple_ml_insights(
                                str(target_for_run),
                                problem_type or "",
                                model_results or {},
                                feature_importances,
                                y_series,
                            )
                            ml_pipeline_status = None
                        except Exception as e:
                            err_text = f"Model training failed: {e}"
                            st.sidebar.error(err_text)
                            ml_pipeline_status = err_text
                    else:
                        st.sidebar.warning(train_msg)
                        ml_pipeline_status = train_msg
                elif target_for_run is not None and target_in_cleaned is None:
                    err_text = (
                        "Could not match the selected target to cleaned column names. "
                        "Try re-uploading the file or pick another column."
                    )
                    st.sidebar.error(err_text)
                    ml_pipeline_status = err_text
                elif target_for_run is None:
                    ml_pipeline_status = (
                        "No target column was selected. Choose a target above, then click "
                        "**Run Full Analysis** again to train models."
                    )

                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = cleaning_report
                st.session_state.summary = summary
                st.session_state.desc = desc
                st.session_state.corr = corr
                st.session_state.model_results = model_results
                st.session_state.feature_importances = feature_importances
                st.session_state.problem_type = problem_type
                st.session_state.target_column = target_for_run
                st.session_state.ml_quick_insights = ml_quick
                st.session_state.ml_training_note = training_note
                st.session_state.ml_pipeline_status = ml_pipeline_status
                st.session_state.analysis_done = True
            st.sidebar.success("Analysis complete.")
            st.rerun()

    # Main content layout: 70/30 split
    left, right = st.columns([0.70, 0.30], gap="large")

    if choice == "dashboard":
        with left:
            panel_upload()
            if df_loaded:
                st.caption(
                    "Open **Data Insights** for schema, correlation views, and charts after you run analysis."
                )

        with right:
            # Forecast / Prediction results (reusing ML outputs)
            _card_open("Forecast / predictions", "Model outcomes for the selected target.")
            mr = st.session_state.get("model_results") or {}
            pt = st.session_state.get("problem_type")
            tgt = st.session_state.get("target_column")
            if mr and pt:
                st.write(f"**Target:** `{tgt}`")
                st.write(f"**Model type:** `{pt}`")
                best = None
                if pt == "classification":
                    best = max(mr.items(), key=lambda x: x[1].get("accuracy", 0))
                    st.caption(f"Best test accuracy (approx.): **{best[1].get('accuracy', 0):.3f}** ({best[0]})")
                else:
                    best = max(mr.items(), key=lambda x: x[1].get("r2_score", -999))
                    st.caption(f"Best test R² (approx.): **{best[1].get('r2_score', 0):.3f}** ({best[0]})")
                st.caption("See full metrics in **Predictive Analysis**.")
            else:
                st.caption("Run Full Analysis with a target selected to populate predictions.")
            _card_close()

            # Anomaly alerts summary (reusing cleaning report)
            _card_open("Anomaly alerts", "Data quality signals from the cleaning pipeline.")
            if st.session_state.get("cleaning_report"):
                rep = st.session_state["cleaning_report"]
                st.write(f"- **Duplicates removed**: {int(rep.get('duplicates_removed', 0) or 0)}")
                before = rep.get("missing_values_before", {}) or {}
                after = rep.get("missing_values_after", {}) or {}
                try:
                    total_before = int(sum(v for v in before.values() if isinstance(v, (int, float))))
                    total_after = int(sum(v for v in after.values() if isinstance(v, (int, float))))
                    st.write(f"- **Missing values (before)**: {total_before}")
                    st.write(f"- **Missing values (after)**: {total_after}")
                except Exception:
                    st.write("- **Missing values**: —")
            else:
                st.caption("Run Full Analysis to see anomaly summary.")
            _card_close()

            _card_open("Quick stats", "At-a-glance dataset signals.")
            if df_loaded:
                st.write(f"- **Rows**: {df.shape[0]}")
                st.write(f"- **Columns**: {df.shape[1]}")
            else:
                st.caption("Upload a dataset to populate quick stats.")
            _card_close()

    elif choice == "insights":
        with left:
            if df_loaded:
                tab_overview, tab_eda, tab_viz = st.tabs(
                    ["Overview", "EDA & stats", "Visualizations"]
                )
                with tab_overview:
                    panel_data_overview(df)
                with tab_eda:
                    panel_eda()
                with tab_viz:
                    panel_visualizations()
            else:
                panel_upload()
        with right:
            if df_loaded:
                _card_open("Quick stats", "Same dataset as in the overview on the left.")
                st.write(f"- **Rows**: {df.shape[0]:,}")
                st.write(f"- **Columns**: {df.shape[1]}")
                st.caption(
                    "Tip: open the **Visualizations** tab to jump straight to charts "
                    "without scrolling through overview and EDA."
                )
                _card_close()
            else:
                st.caption("Upload a dataset to explore it here.")

    elif choice == "predictive":
        with left:
            panel_ml()
        with right:
            panel_data_overview(df) if df_loaded else panel_upload()

    elif choice == "anomalies":
        with left:
            panel_cleaning_report()
        with right:
            panel_data_overview(df) if df_loaded else panel_upload()

    elif choice == "ai_insights":
        with left:
            panel_ai_insights()
        with right:
            panel_data_overview(df) if df_loaded else panel_upload()

    elif choice == "reports":
        with left:
            panel_download()
        with right:
            panel_data_overview(df) if df_loaded else panel_upload()


if __name__ == "__main__":
    main()
