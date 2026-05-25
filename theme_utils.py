"""Shared light/dark theme helpers for CricScope Streamlit pages."""

import streamlit as st

THEME_KEY = "theme"


def init_theme(default: str = "dark") -> str:
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = default
    return st.session_state[THEME_KEY]


def render_theme_toggle() -> str:
    theme = init_theme()
    st.markdown('<div class="sidebar-section-label">Appearance</div>', unsafe_allow_html=True)
    choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if theme == "dark" else 1,
        key="theme_radio",
        label_visibility="collapsed",
    )
    st.session_state[THEME_KEY] = choice.lower()
    return st.session_state[THEME_KEY]


def theme_stylesheet(mode: str) -> str:
    if mode == "light":
        return """
<style>
[data-testid="stAppViewContainer"] {
    background: #f4efe3 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(180,140,60,0.08) 0%, transparent 50%) !important;
}
html, body, [class*="css"], .stApp { color: #2c2820 !important; }
section[data-testid="stSidebar"] {
    background: #ebe6da !important;
    border-right: 1px solid rgba(154,123,26,0.2) !important;
}
.sidebar-logo-text {
    background: linear-gradient(135deg, #8a6f12 0%, #b8941f 50%, #7a5f10 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
}
.sidebar-tagline, .sidebar-section-label { color: rgba(120,95,30,0.65) !important; }
.stButton > button { color: rgba(60,50,30,0.75) !important; }
.stButton > button:hover { color: #8a6f12 !important; background: rgba(212,175,55,0.12) !important; }
.hero-title {
    background: linear-gradient(160deg, #3d3528 0%, #6b5a28 40%, #9a7b1a 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
}
.hero-subtitle { color: rgba(60,50,35,0.65) !important; }
.hero-eyebrow { color: rgba(120,95,30,0.7) !important; }
.hero-wrapper { border-bottom-color: rgba(154,123,26,0.15) !important; }
.stat-pill {
    background: rgba(255,255,255,0.7) !important;
    border-color: rgba(154,123,26,0.2) !important;
}
.stat-value { color: #7a6218 !important; }
.stat-label { color: rgba(80,70,50,0.5) !important; }
.stats-row { border-bottom-color: rgba(154,123,26,0.12) !important; }
.input-card {
    background: rgba(255,255,255,0.75) !important;
    border-color: rgba(154,123,26,0.2) !important;
}
section[data-testid="stMain"] [data-testid="stSelectbox"] > div > div,
section[data-testid="stMain"] [data-testid="stNumberInput"] > div > div > input,
section[data-testid="stMain"] [data-testid="stSlider"] > div {
    background: rgba(255,255,255,0.9) !important;
    border-color: rgba(154,123,26,0.25) !important;
    color: #2c2820 !important;
}
section[data-testid="stMain"] [data-testid="stSelectbox"] label,
section[data-testid="stMain"] [data-testid="stNumberInput"] label,
section[data-testid="stMain"] [data-testid="stSlider"] label { color: rgba(80,70,50,0.65) !important; }
.fixture-card {
    background: rgba(255,255,255,0.8) !important;
    border-color: rgba(154,123,26,0.2) !important;
}
.fixture-role { color: rgba(80,70,50,0.45) !important; }
.fixture-vs { color: rgba(154,123,26,0.35) !important; }
.prediction-card {
    background: rgba(255,255,255,0.85) !important;
    border-color: rgba(154,123,26,0.3) !important;
}
/* Stats page */
.stats-theme-body { color: #2c2820; }
.stats-theme-body .glass-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(154,123,26,0.28) !important;
}
.stats-theme-body .section-title,
.stats-theme-body h1, .stats-theme-body h2, .stats-theme-body h3 { color: #8a6f12 !important; }
.stats-theme-body .gold { color: #8a6f12 !important; }
.stats-theme-body .stat-pill {
    background: rgba(212,175,55,0.15) !important;
    border-color: rgba(154,123,26,0.35) !important;
    color: #7a6218 !important;
}
.stats-theme-body .win-bar-wrap { background: rgba(0,0,0,0.06) !important; }
.stats-theme-body .stats-subtitle { color: #6b6358 !important; }
.stats-theme-body .stats-muted { color: #7a7268 !important; }
.stats-theme-body .stats-footer { color: #9a9288 !important; }
.stats-theme-body .stats-card-value { color: #8a6f12 !important; }
.stats-theme-body .stats-venue-text { color: #2c2820 !important; }
.stats-theme-body .stats-rank-muted { color: #7a7268 !important; }
.stats-theme-body .stats-player-name { color: #2c2820 !important; }
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.95) !important;
    color: #2c2820 !important;
    border-color: rgba(154,123,26,0.25) !important;
}
div[data-testid="stDataFrame"] { border: 1px solid rgba(154,123,26,0.2) !important; }
</style>
"""
    return """
<style>
[data-testid="stAppViewContainer"] {
    background: #080808 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%) !important;
}
.stats-theme-body .stats-subtitle { color: #888 !important; }
.stats-theme-body .stats-muted { color: #aaa !important; }
.stats-theme-body .stats-footer { color: #444 !important; }
.stats-theme-body .stats-card-value { color: #d4af37 !important; }
.stats-theme-body .stats-card-value-muted { color: #888 !important; }
.stats-theme-body .stats-venue-text { color: #e8e0d0 !important; }
.stats-theme-body .stats-rank-muted { color: #888 !important; }
.stats-theme-body .stats-player-name { color: #e8e0d0 !important; }
</style>
"""
