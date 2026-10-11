"""Shared visual theme for the Streamlit app.

Purely presentational -- CSS injection plus small HTML-building helpers.
Nothing here touches data, session_state keys, widget keys, or business
logic; it is safe to import from every page without risking the app's
actual behavior (in particular the Safer Route Planner's pin/widget logic,
which was hard-won and must not be touched by styling changes).

Design language: navy/blue "municipal decision-support" palette, white
content cards on a soft slate background, rounded corners, subtle shadows,
Inter typeface. Risk-status colors are fixed to the semantic convention
used throughout this project: green = low, orange = medium, red = high.
"""

from __future__ import annotations

import streamlit as st

# --- Palette ------------------------------------------------------------------
# Kept as plain constants (not a dict) so `from src.app.theme import RISK_HIGH`
# etc. is easy to reuse anywhere a hex value is needed outside CSS (e.g. a
# plotly color map).
NAVY_900 = "#0B1E3D"
NAVY_800 = "#122A4D"
NAVY_700 = "#1B3660"
BLUE_600 = "#2563EB"
BLUE_500 = "#3B82F6"
SLATE_50 = "#F5F7FA"
SLATE_100 = "#EEF1F6"
SLATE_200 = "#E2E8F0"
SLATE_500 = "#64748B"
SLATE_600 = "#475569"
SLATE_900 = "#0F172A"
WHITE = "#FFFFFF"

RISK_LOW = "#16A34A"
RISK_MEDIUM = "#F59E0B"
RISK_HIGH = "#DC2626"
RISK_NEUTRAL = "#94A3B8"  # "no damage" / unknown

RISK_COLORS = {"low": RISK_LOW, "medium": RISK_MEDIUM, "high": RISK_HIGH}
RISK_LABELS = {"low": "Low risk", "medium": "Medium risk", "high": "High risk"}


def inject_custom_css() -> None:
    """Inject the app-wide stylesheet. Call once per page (idempotent --
    Streamlit re-renders this <style> block harmlessly on every rerun).
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --navy-900: {NAVY_900};
            --navy-800: {NAVY_800};
            --navy-700: {NAVY_700};
            --blue-600: {BLUE_600};
            --blue-500: {BLUE_500};
            --slate-50: {SLATE_50};
            --slate-100: {SLATE_100};
            --slate-200: {SLATE_200};
            --slate-500: {SLATE_500};
            --slate-600: {SLATE_600};
            --slate-900: {SLATE_900};
            --risk-low: {RISK_LOW};
            --risk-medium: {RISK_MEDIUM};
            --risk-high: {RISK_HIGH};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        /* --- Page background & layout ------------------------------------------- */
        [data-testid="stAppViewContainer"] {{
            background: var(--slate-50);
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stMain"] .block-container,
        section.main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}

        /* --- Typography ----------------------------------------------------------- */
        h1, h2, h3 {{
            color: var(--navy-900);
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        h1 {{ font-size: 1.9rem; }}
        h2, [data-testid="stSubheader"] {{ font-size: 1.3rem; }}
        p, li, label, [data-testid="stCaptionContainer"] {{
            color: var(--slate-600);
        }}

        /* --- Sidebar ---------------------------------------------------------------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--navy-900) 0%, var(--navy-800) 100%);
            border-right: 1px solid var(--navy-700);
        }}
        section[data-testid="stSidebar"] * {{
            color: #E7ECF6;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #FFFFFF;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.12);
        }}
        /* The global [data-testid="stMetric"] rule below gives metrics a
           white card background -- inside the navy sidebar that needs its
           own background (not white, which would clash) so the light text
           stays readable instead of turning into white-on-white. */
        section[data-testid="stSidebar"] [data-testid="stMetric"] {{
            background: var(--navy-700);
            border: 1px solid rgba(255,255,255,0.14);
        }}
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
            color: #FFFFFF;
            font-size: 1.3rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
            color: #A9B7D0;
        }}
        section[data-testid="stSidebar"] [data-testid="stAlert"] {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: var(--blue-600);
            color: #FFFFFF;
            border: none;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: var(--blue-500);
        }}
        section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
            background-color: var(--blue-500);
        }}

        /* Page navigation links */
        [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
            margin: 1px 0;
            transition: background 0.15s ease;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255,255,255,0.08);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: var(--blue-600);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] span {{
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        /* --- Metric cards ------------------------------------------------------------ */
        [data-testid="stMetric"] {{
            background: var(--white, #FFFFFF);
            border: 1px solid var(--slate-200);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.04);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--slate-500);
            font-weight: 500;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--navy-900);
            font-weight: 700;
        }}

        /* --- Cards: expanders & containers used as panels ------------------------------ */
        [data-testid="stExpander"] {{
            background: #FFFFFF;
            border: 1px solid var(--slate-200);
            border-radius: 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{
            font-weight: 600;
            color: var(--navy-900);
            padding: 0.6rem 0.9rem;
        }}

        /* --- Buttons ------------------------------------------------------------------ */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid var(--slate-200);
            transition: all 0.15s ease;
        }}
        .stButton > button[kind="primary"] {{
            background: var(--blue-600);
            border-color: var(--blue-600);
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--navy-800);
            border-color: var(--navy-800);
        }}
        .stButton > button:hover {{
            border-color: var(--blue-500);
            color: var(--blue-600);
        }}

        /* --- Alerts (info / success / warning / error) --------------------------------- */
        [data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--slate-200);
            padding: 0.85rem 1rem;
        }}

        /* --- Inputs --------------------------------------------------------------------- */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] {{
            border-radius: 10px !important;
        }}

        /* --- File uploader ---------------------------------------------------------------- */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--slate-50);
            border: 1.5px dashed var(--slate-200);
            border-radius: 14px;
        }}

        /* --- Tables / dataframes ------------------------------------------------------------ */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--slate-200);
        }}

        /* --- Tabs (styled for reuse even where not yet used) --------------------------------- */
        [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid var(--slate-200);
        }}
        [data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }}

        /* --- Maps (st_folium renders inside an iframe) ---------------------------------------- */
        iframe {{
            border-radius: 14px;
            border: 1px solid var(--slate-200);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 16px rgba(15, 23, 42, 0.05);
        }}

        /* --- Page header banner ---------------------------------------------------------------- */
        .app-page-header {{
            background: linear-gradient(120deg, var(--navy-900) 0%, var(--navy-700) 100%);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(11, 30, 61, 0.18);
        }}
        .app-page-header .app-page-title {{
            color: #FFFFFF;
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .app-page-header .app-page-subtitle {{
            color: #C7D2E8;
            font-size: 0.95rem;
            margin-top: 0.3rem;
        }}

        /* --- Risk badges ------------------------------------------------------------------------- */
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.82rem;
            color: #FFFFFF;
        }}

        /* --- Hero image (landing page) ------------------------------------------------------------- */
        [data-testid="stImage"] img {{
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(11, 30, 61, 0.22);
        }}

        /* --- Home stat / step cards -------------------------------------------------------------------- */
        /* Fixed min-heights (rather than relying on flex-stretch across
           Streamlit's column DOM, which isn't consistent) so every card in a
           row lines up regardless of how much its text wraps. */
        .stat-card, .step-card {{
            background: #FFFFFF;
            border: 1px solid var(--slate-200);
            border-radius: 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            display: flex;
            flex-direction: column;
        }}
        .stat-card {{
            padding: 1.1rem 1.2rem;
            min-height: 108px;
            justify-content: center;
            border-top: 3px solid var(--accent, var(--blue-600));
        }}
        .stat-card .stat-value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--navy-900);
            line-height: 1.2;
        }}
        .stat-card .stat-label {{
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--slate-900);
            margin-top: 0.15rem;
        }}
        .stat-card .stat-sublabel {{
            font-size: 0.78rem;
            color: var(--slate-500);
            margin-top: 0.1rem;
        }}
        .step-card {{
            padding: 1.1rem 1.2rem;
            min-height: 168px;
        }}
        .step-card .step-number {{
            background: var(--blue-600);
            color: #FFFFFF;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-bottom: 0.65rem;
        }}
        .step-card .step-title {{
            font-weight: 700;
            color: var(--navy-900);
            margin-bottom: 0.3rem;
            font-size: 0.98rem;
        }}
        .step-card .step-desc {{
            font-size: 0.82rem;
            color: var(--slate-500);
            line-height: 1.4;
        }}
        .step-chevron {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 168px;
            color: var(--slate-200);
            font-size: 1.4rem;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card_html(value: str, label: str, sublabel: str, accent: str) -> str:
    """Compact number card for the landing page's summary row. No icon --
    a colored top border ties it to the accent color instead."""
    return f"""
    <div class="stat-card" style="--accent:{accent};">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-sublabel">{sublabel}</div>
    </div>
    """


def step_card_html(number: int, title: str, description: str) -> str:
    """One step in the landing page's "how it works" strip. No icon --
    just the numbered badge, title, and description."""
    return f"""
    <div class="step-card">
        <span class="step-number">{number}</span>
        <div class="step-title">{title}</div>
        <div class="step-desc">{description}</div>
    </div>
    """


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Consistent navy banner header, replacing bare st.title/st.caption pairs."""
    st.markdown(
        f"""
        <div class="app-page-header">
            <p class="app-page-title">{icon} {title}</p>
            <p class="app-page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(category: str) -> str:
    """Small colored pill for a risk category -- e.g. for embedding in a
    popup or a markdown line. Does not render by itself; wrap in
    st.markdown(..., unsafe_allow_html=True)."""
    color = RISK_COLORS.get(category, RISK_NEUTRAL)
    label = RISK_LABELS.get(category, category.title() if category else "Unknown")
    return f'<span class="risk-badge" style="background:{color};">{label}</span>'
