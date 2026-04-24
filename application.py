import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="CricScope", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# -----------------------------------
# LUXURY CSS
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ---- RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2dfd8;
}

[data-testid="stAppViewContainer"] {
    background: #080808;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%);
    min-height: 100vh;
}

/* Hide only Streamlit branding — leave header & sidebar toggle untouched */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid rgba(212,175,55,0.12);
    width: 260px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
}

.sidebar-brand {
    padding: 36px 28px 24px;
    border-bottom: 1px solid rgba(212,175,55,0.1);
    margin-bottom: 16px;
}

.sidebar-logo-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin-bottom: 4px;
}

.sidebar-tagline {
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.45);
    font-weight: 400;
}

.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.2), transparent);
    margin: 8px 0;
}

.sidebar-section-label {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(180,160,100,0.35);
    padding: 12px 28px 6px;
    font-weight: 500;
}

/* ---- NAV BUTTONS ---- */
.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 0;
    color: rgba(220,210,180,0.65);
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0.5px;
    padding: 11px 28px;
    height: auto;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background: rgba(212,175,55,0.06);
    color: #d4af37;
    border: none;
    box-shadow: none;
}

.stButton > button:active,
.stButton > button:focus {
    background: rgba(212,175,55,0.1);
    color: #f0d060;
    border: none;
    box-shadow: none;
    outline: none;
}

/* ---- MAIN CONTENT AREA ---- */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ---- HERO SECTION ---- */
.hero-wrapper {
    padding: 64px 60px 40px;
    border-bottom: 1px solid rgba(212,175,55,0.08);
    position: relative;
    overflow: hidden;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px; right: -60px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(212,175,55,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: 18px;
    font-weight: 400;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(52px, 7vw, 88px);
    font-weight: 600;
    line-height: 0.95;
    letter-spacing: -1px;
    background: linear-gradient(160deg, #ffffff 0%, #f8f0d0 30%, #d4af37 70%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 18px;
}

.hero-subtitle {
    font-size: 15px;
    color: rgba(220,210,185,0.55);
    font-weight: 300;
    letter-spacing: 0.3px;
    max-width: 460px;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-size: 11px;
    color: rgba(212,175,55,0.8);
    letter-spacing: 0.5px;
    margin-bottom: 24px;
    width: fit-content;
}

.hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #d4af37;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ---- STAT PILLS ---- */
.stats-row {
    display: flex;
    gap: 16px;
    padding: 24px 60px;
    border-bottom: 1px solid rgba(212,175,55,0.06);
}

.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 22px;
    transition: all 0.25s ease;
}

.stat-pill:hover {
    background: rgba(212,175,55,0.04);
    border-color: rgba(212,175,55,0.15);
    transform: translateY(-1px);
}

.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    color: #e8d89a;
    line-height: 1;
    margin-bottom: 6px;
}

.stat-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.4);
}

/* ---- ANALYSIS SECTION ---- */
.section-header {
    padding: 40px 60px 0;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 500;
    color: #f0e8cc;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.section-desc {
    font-size: 13px;
    color: rgba(200,185,140,0.4);
    letter-spacing: 0.3px;
}

/* ---- INPUT CARD ---- */
.input-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s ease;
}

.input-card:hover {
    border-color: rgba(212,175,55,0.15);
}

.input-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: 12px;
    font-weight: 500;
}

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}

.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
}

/* Slider track */
.stSlider [data-testid="stSlider"] > div {
    background: rgba(212,175,55,0.15) !important;
}

.stSlider [data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, #d4af37, #f0d060) !important;
}

/* ---- TEAM VS CARD ---- */
.team-vs-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 36px 28px;
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.team-vs-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(212,175,55,0.04) 0%, transparent 60%);
    pointer-events: none;
}

.team-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 3px;
    margin-top: 14px;
}

.vs-divider {
    font-family: 'Cormorant Garamond', serif;
    font-size: 48px;
    font-weight: 300;
    color: rgba(212,175,55,0.25);
    line-height: 1;
    letter-spacing: -2px;
}

.team-logo-glow {
    border-radius: 50%;
    transition: box-shadow 0.3s ease;
    width: 90px;
    height: 90px;
    object-fit: contain;
}

/* ---- ANALYZE BUTTON ---- */
.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 40%, #e8c84a 100%);
    color: #0a0800;
    border: none;
    border-radius: 14px;
    height: 52px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(212,175,55,0.2), 0 0 0 0 rgba(212,175,55,0);
    width: 100%;
}

.stButton.analyze-btn > button:hover {
    box-shadow: 0 12px 48px rgba(212,175,55,0.35), 0 0 60px rgba(212,175,55,0.1);
    transform: translateY(-2px);
    filter: brightness(1.05);
    color: #0a0800;
    border: none;
}

/* ---- PREDICTION CARD ---- */
.prediction-card {
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 24px;
    padding: 36px 32px;
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4af37, transparent);
}

.prediction-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(212,175,55,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.prediction-label {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.4);
    margin-bottom: 24px;
    font-weight: 500;
}

.win-team-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px;
    font-weight: 600;
    color: #f0e0a0;
    line-height: 1;
    margin-bottom: 8px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: 72px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}

.win-prob-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.35);
    margin-bottom: 28px;
}

/* ---- PROGRESS BAR CUSTOM ---- */
.prob-bar-wrapper {
    position: relative;
    margin: 20px 0 14px;
}

.prob-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #b8962e, #d4af37, #f0d060);
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 12px rgba(212,175,55,0.4);
}

.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 11px;
    color: rgba(200,185,140,0.4);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}

/* ---- METRICS ROW ---- */
.metrics-row {
    display: flex;
    gap: 10px;
    margin-top: 18px;
}

.metric-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
}

.metric-chip-value {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    color: #d4c080;
    font-weight: 500;
    margin-bottom: 4px;
}

.metric-chip-label {
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(180,165,115,0.35);
}

/* ---- STRAY STREAMLIT COMPONENTS ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #b8962e, #d4af37) !important;
    border-radius: 100px !important;
}

.stProgress > div {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 100px !important;
    height: 6px !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px;
}

div[data-testid="metric-container"] label {
    color: rgba(200,185,140,0.45) !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #e8d89a !important;
    font-size: 28px !important;
}

/* ---- SEPARATOR ---- */
hr {
    border: none;
    border-top: 1px solid rgba(212,175,55,0.08);
    margin: 0;
}

/* ---- CONTENT PADDING ---- */
.main-pad {
    padding: 0 60px 60px;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }

/* ---- PROFILE CARD LINKS ---- */
.profile-link {
    display: flex; align-items: center; gap: 10px;
    text-decoration: none; padding: 8px 10px;
    border-radius: 9px; margin-bottom: 4px;
    background: transparent;
    transition: background 0.2s ease;
}
.profile-link:hover { background: rgba(212,175,55,0.07); }
.profile-link span {
    font-size: 11px; color: rgba(200,185,140,0.55);
    font-weight: 400; letter-spacing: 0.2px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    transition: color 0.2s;
}
.profile-link:hover span { color: rgba(212,175,55,0.8); }
.profile-link svg { flex-shrink: 0; }

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TEAM DATA
# -----------------------------------
team_data = {
    "Chennai Super Kings": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg",
        "abbr": "CSK", "color": "#facc15"
    },
    "Delhi Capitals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg",
        "abbr": "DC", "color": "#3b82f6"
    },
    "Punjab Kings": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg",
        "abbr": "PBKS", "color": "#ef4444"
    },
    "Kolkata Knight Riders": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg",
        "abbr": "KKR", "color": "#7c3aed"
    },
    "Mumbai Indians": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg",
        "abbr": "MI", "color": "#3b82f6"
    },
    "Rajasthan Royals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg",
        "abbr": "RR", "color": "#ec4899"
    },
    "Royal Challengers Bangalore": {
        "logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg",
        "abbr": "RCB", "color": "#dc2626"
    },
    "Sunrisers Hyderabad": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg",
        "abbr": "SRH", "color": "#f97316"
    }
}

# -----------------------------------
# MODEL
# -----------------------------------
@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    df['over'] = df['over'].replace(0, 0.1)

    df['crr'] = df['current_score'] / (df['over'] + df['ball'] / 6)
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)
    return pipe

pipe = train_model()

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <span class="sidebar-logo-text">CRICSCOPE</span>
            <span class="sidebar-tagline">Match Intelligence Platform</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    if st.button("◈  Dashboard", key="nav_dash"):
        st.session_state.page = "Dashboard"

    if st.button("◉  Match Analysis", key="nav_analysis"):
        st.session_state.page = "Analysis"

    st.markdown('<div style="height:1px; background:rgba(212,175,55,0.08); margin:16px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Built By</div>', unsafe_allow_html=True)

    # Profile card — rendered as separate st.markdown blocks to stay within Streamlit's HTML allowlist
    st.markdown("""
        <div style="padding:0 18px 8px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);
                        border-radius:16px;padding:20px 18px 14px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:60px;
                            background:radial-gradient(ellipse at 50% 0%,rgba(212,175,55,0.08) 0%,transparent 70%);
                            pointer-events:none;"></div>
                <div style="width:44px;height:44px;border-radius:50%;
                            background:linear-gradient(135deg,#c9a227,#f0d060);
                            display:flex;align-items:center;justify-content:center;
                            font-size:16px;font-weight:700;color:#0a0800;
                            margin-bottom:12px;box-shadow:0 0 18px rgba(212,175,55,0.25);">AS</div>
                <div style="font-size:17px;font-weight:600;color:#f0e8cc;
                            letter-spacing:0.5px;margin-bottom:3px;">Arnav Singh</div>
                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                            color:rgba(212,175,55,0.4);margin-bottom:18px;font-weight:500;">ML · Data · Analytics</div>
                <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,0.15),transparent);margin-bottom:12px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Links rendered as st.markdown with href — Streamlit allows plain <a> with href+target
    st.markdown("""
        <div style="padding:0 18px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);
                        border-top:none;border-radius:0 0 16px 16px;padding:4px 10px 14px;">
                <p style="margin:0 0 2px 0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">✉</span>
                    <a href="mailto:itsarnav.singh80@gmail.com"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        itsarnav.singh80@gmail.com
                    </a>
                </p>
                <p style="margin:0 0 2px 0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">in</span>
                    <a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        linkedin.com/in/arnav-singh
                    </a>
                </p>
                <p style="margin:0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">&#9670;</span>
                    <a href="https://github.com/Arnav-Singh-5080" target="_blank"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        Arnav-Singh-5080
                    </a>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center;margin-top:16px;padding-bottom:24px;font-size:9px;
                    letter-spacing:1.5px;text-transform:uppercase;color:rgba(200,185,140,0.18);">
            CricScope v2.0 · IPL Edition
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------
if st.session_state.page == "Dashboard":

    st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-eyebrow">IPL Match Intelligence · Season 2025</div>
            <div class="hero-badge">
                <div class="hero-dot"></div>
                Live Predictions Active
            </div>
            <div class="hero-title">CricScope</div>
            <div class="hero-subtitle">
                Precision match analytics engineered for modern cricket.
                Real-time win probability powered by machine learning.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-row">
            <div class="stat-pill">
                <div class="stat-value">8</div>
                <div class="stat-label">IPL Teams</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">ML</div>
                <div class="stat-label">Model Type</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">120</div>
                <div class="stat-label">Balls Tracked</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">6+</div>
                <div class="stat-label">Key Signals</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="padding: 48px 60px;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:13px; letter-spacing:3px;
                        text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom:28px;">
                IPL Teams
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:12px;">
    """, unsafe_allow_html=True)

    team_cols = st.columns(4)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        with team_cols[i % 4]:
            st.markdown(f"""
                <div style="
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;
                    padding:20px;
                    text-align:center;
                    transition:all 0.25s ease;
                    margin-bottom:12px;
                ">
                    <div style="width:72px;height:72px;border-radius:50%;margin:0 auto;
                                overflow:hidden;background:#111;
                                box-shadow:0 0 20px {tdata['color']}50;
                                display:flex;align-items:center;justify-content:center;">
                        <img src="{tdata['logo']}"
                             style="width:100%;height:100%;object-fit:cover;
                                    mix-blend-mode:screen;border-radius:50%;" />
                    </div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:18px; font-weight:600;
                                color:{tdata['color']}; letter-spacing:2px; margin-top:12px;">
                        {tdata['abbr']}
                    </div>
                    <div style="font-size:10px; color:rgba(200,185,140,0.35); margin-top:4px;
                                letter-spacing:0.5px;">
                        {team_name}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div style="padding:0 60px 32px; text-align:center;">
            <div style="display:inline-block; background:rgba(212,175,55,0.06); border:1px solid rgba(212,175,55,0.15);
                        border-radius:14px; padding:20px 36px;">
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:rgba(212,175,55,0.5);margin-bottom:8px;">Get Started</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;color:#f0e8cc;font-weight:500;">
                    Open Match Analysis from the sidebar →
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def generate_over_progression(target, score, wickets, balls_left, batting_team, bowling_team, pipe):
    data = []

    temp_score = score
    temp_wickets = 10 - wickets
    temp_balls = balls_left

    for i in range(0, balls_left, 6):
        if temp_balls <= 0 or temp_wickets <= 0:
            break

        runs_left = target - temp_score

        if runs_left <= 0:
            break

        overs_completed = (120 - temp_balls) // 6

        crr = temp_score / (overs_completed + 1e-5)
        rrr = (runs_left * 6) / temp_balls

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': ['Mumbai'],
            'runs_left': [runs_left],
            'balls_left': [temp_balls],
            'wickets': [temp_wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        prob = pipe.predict_proba(input_df)[0][1] * 100

        data.append({
            "over": overs_completed,
            "win_prob": prob,
            "runs_left": runs_left,
            "wickets": temp_wickets
        })

        # simulate next over
        temp_balls -= 6
        temp_score += np.random.randint(6, 15)

        if np.random.rand() < 0.15:
            temp_wickets -= 1

    return pd.DataFrame(data)

# -----------------------------------
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    st.markdown("""
        <div class="hero-wrapper" style="padding-bottom:32px;">
            <div class="hero-eyebrow">Win Probability Engine</div>
            <div class="hero-title" style="font-size:clamp(36px,4vw,56px); margin-bottom:10px;">Match Analysis</div>
            <div class="hero-subtitle">Configure the match state below to compute real-time win probabilities.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())

    # ---- INPUT SECTION ----
    st.markdown("""
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:rgba(212,175,55,0.4);margin-bottom:20px;font-weight:500;">
            Match Configuration
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Teams</div>', unsafe_allow_html=True)
        batting_team = st.selectbox("Batting Team", teams, key="bat")
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowl")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Match State</div>', unsafe_allow_html=True)
        target = st.number_input("Target Score", min_value=50, max_value=300, value=180, step=1)
        score = st.number_input("Current Score", min_value=0, max_value=target - 1, value=50, step=1)
        col_ov, col_wk = st.columns(2)
        with col_ov:
            overs = st.slider("Overs Completed", min_value=1, max_value=19, value=10)
        with col_wk:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    # ---- TEAM VS DISPLAY ----
    t1 = team_data[batting_team]
    if bowling_team in team_data:
        t2 = team_data[bowling_team]
    else:
        t2 = team_data[teams[1]]

    st.markdown("""
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">
            Fixture
        </div>
    """, unsafe_allow_html=True)

    vs_col1, vs_col2, vs_col3 = st.columns([2, 1, 2])

    with vs_col1:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px;padding:28px;text-align:center;
                        box-shadow:0 0 40px {t1['color']}12;">
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t1['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t1['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:600;
                            color:{t1['color']};letter-spacing:3px;margin-top:14px;">
                    {t1['abbr']}
                </div>
                <div style="font-size:10px;color:rgba(200,185,140,0.3);margin-top:4px;letter-spacing:0.5px;">
                    BATTING
                </div>
            </div>
        """, unsafe_allow_html=True)

    with vs_col2:
        st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        font-family:'Cormorant Garamond',serif;font-size:52px;font-weight:300;
                        color:rgba(212,175,55,0.2);letter-spacing:-2px;padding:28px 0;">
                vs
            </div>
        """, unsafe_allow_html=True)

    with vs_col3:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px;padding:28px;text-align:center;
                        box-shadow:0 0 40px {t2['color']}12;">
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t2['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t2['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:600;
                            color:{t2['color']};letter-spacing:3px;margin-top:14px;">
                    {t2['abbr']}
                </div>
                <div style="font-size:10px;color:rgba(200,185,140,0.3);margin-top:4px;letter-spacing:0.5px;">
                    BOWLING
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    # ---- ANALYZE BUTTON ----
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze = st.button("Run Analysis", key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- PREDICTION OUTPUT ----
    if analyze:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': ['Mumbai'],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        with st.spinner(""):
            time.sleep(0.4)
            proba = pipe.predict_proba(input_df)[0]

        win = proba[1]
        lose = proba[0]

        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                        color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">
                Prediction Output
            </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2, gap="large")

        with res_col1:
            bat_pct = round(win * 100)
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Batting Team · {t1['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
                        {batting_team}
                    </div>
                    <div class="win-probability">{bat_pct}%</div>
                    <div class="win-prob-label">Win Probability</div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width:{bat_pct}%;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bat_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{score}</div>
                            <div class="metric-chip-label">Score</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{runs_left}</div>
                            <div class="metric-chip-label">Needed</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{balls_left}</div>
                            <div class="metric-chip-label">Balls Left</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            bowl_pct = round(lose * 100)
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                            border-radius:24px;padding:36px 32px;position:relative;overflow:hidden;">
                    <div class="prediction-label">Bowling Team · {t2['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
                        {bowling_team}
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:72px;font-weight:500;
                                color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">
                        {bowl_pct}%
                    </div>
                    <div class="win-prob-label">Win Probability</div>
                    <div class="prob-bar-track">
                        <div style="height:100%;border-radius:100px;
                                    background:rgba(200,185,140,0.2);
                                    width:{bowl_pct}%;transition:width 0.8s ease;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bowl_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(crr, 2)}</div>
                            <div class="metric-chip-label">CRR</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(rrr, 2)}</div>
                            <div class="metric-chip-label">RRR</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{10 - wickets}</div>
                            <div class="metric-chip-label">In Hand</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ---- SUMMARY ROW ----
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        verdict = batting_team if win > 0.5 else bowling_team
        conf = max(win, lose)
        conf_label = "High" if conf > 0.75 else "Moderate" if conf > 0.55 else "Close"

        st.markdown(f"""
            <div style="background:rgba(212,175,55,0.03);border:1px solid rgba(212,175,55,0.1);
                        border-radius:16px;padding:20px 28px;display:flex;
                        align-items:center;justify-content:space-between;">
                <div>
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">Model Verdict</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#f0e8cc;">
                        {verdict} favoured to win
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">Confidence</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;color:#d4af37;">
                        {conf_label} · {round(conf*100)}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ---- WIN PROBABILITY CHART ----
    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">
            Win Probability Progression
        </div>
    """, unsafe_allow_html=True)

    df_prog = generate_over_progression(
        target, score, wickets, balls_left,
        batting_team, bowling_team, pipe
    )

    if not df_prog.empty:
        fig = go.Figure()

        # Smoother curve (spline)
        fig.add_trace(go.Scatter(
            x=df_prog['over'],
            y=df_prog['win_prob'],
            mode='lines+markers',
            line=dict(color='#d4af37', width=3, shape='spline', smoothing=1.3),
            name=f'{batting_team} Win Probability',
            customdata=df_prog[['runs_left', 'wickets']],
            hovertemplate=
                '<b>Over:</b> %{x}<br>' +
                '<b>Win %:</b> %{y:.2f}<br>' +
                '<b>Runs Left:</b> %{customdata[0]}<br>' +
                '<b>Wickets:</b> %{customdata[1]}<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(212,175,55,0.08)'
        ))

        # Dual team probability line
        fig.add_trace(go.Scatter(
            x=df_prog['over'],
            y=100 - df_prog['win_prob'],
            mode='lines',
            line=dict(color='rgba(180,180,180,0.5)', width=2, dash='dot', shape='spline', smoothing=1.3),
            name=f'{bowling_team} Win Probability',
            hovertemplate='<b>Opponent Win %:</b> %{y:.2f}<extra></extra>'
        ))

        # Wicket markers
        df_prog['wicket_fall'] = df_prog['wickets'].diff().fillna(0) < 0
        wickets_df = df_prog[df_prog['wicket_fall']]
        if not wickets_df.empty:
            fig.add_trace(go.Scatter(
                x=wickets_df['over'],
                y=wickets_df['win_prob'],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                name='Wicket',
                hovertemplate='<b>WICKET!</b><br>Over: %{x}<extra></extra>'
            ))

        # Death overs zone (vertical highlight)
        fig.add_vrect(
            x0=15, x1=20,
            fillcolor="rgba(255,0,0,0.05)",
            layer="below",
            line_width=0
        )

        # Dynamic title
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#080808',
            paper_bgcolor='#080808',
            font=dict(color='#e2dfd8'),
            xaxis_title='Overs',
            yaxis_title='Win Probability (%)',
            height=400,
            title=dict(
                text=f"{batting_team} Win Probability Curve",
                font=dict(size=20, family='Cormorant Garamond, serif'),
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close main-pad
