import streamlit as st

# BASE_CSS contains styles shared across all pages of CricScope
BASE_CSS = """
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
    width: 300px !important;
    min-width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }

/* ---- MAIN CONTENT AREA ---- */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ---- GLASS CARD ---- */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(212,175,55,0.15);
}

/* ---- HERO SECTION BASE ---- */
.hero-wrapper {
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
    font-weight: 400;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    line-height: 0.95;
    letter-spacing: -1px;
    background: linear-gradient(160deg, #ffffff 0%, #f8f0d0 30%, #d4af37 70%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 15px;
    color: rgba(220,210,185,0.55);
    font-weight: 300;
    letter-spacing: 0.3px;
    max-width: 560px;
    line-height: 1.6;
}

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}

.stSelectbox label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
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
"""

# DASHBOARD_CSS contains styles specific to the main page (application.py)
DASHBOARD_CSS = """
/* ---- SIDEBAR PREMIUM GLASSMORPHISM ---- */
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] a:visited,
section[data-testid="stSidebar"] a:hover,
section[data-testid="stSidebar"] a:active {
    text-decoration: none !important;
    color: inherit !important;
}

.sidebar-brand {
    padding: 40px 32px 28px;
    border-bottom: 1px solid rgba(212,175,55,0.1);
    margin-bottom: 20px;
}

.sidebar-logo-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 600;
    letter-spacing: 3.5px;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin-bottom: 6px;
}

.sidebar-tagline {
    font-size: 11px;
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
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(180,160,100,0.35);
    padding: 14px 32px 8px;
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
    font-size: 14px;
    font-weight: 400;
    letter-spacing: 0.5px;
    padding: 13px 32px;
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

/* ---- HERO OVERRIDES & EXTRA ---- */
.hero-wrapper {
    padding: 64px 60px 40px;
}

.hero-eyebrow {
    margin-bottom: 18px;
}

.hero-title {
    font-size: clamp(52px, 7vw, 88px);
    margin-bottom: 18px;
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

/* ---- ST INPUT OVERRIDES EXTENSION ---- */
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}

.stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
}

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

.main-pad {
    padding: 0 60px 60px;
}

/* ---- CONFUSION MATRIX ---- */
.matrix-wrapper {
    background: rgba(255,255,255,0.01);
    border: 1px solid rgba(212,175,55,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-top: 20px;
}

.matrix-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.45);
    margin-bottom: 16px;
    text-align: center;
    font-weight: 500;
}

.matrix-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 10px;
}

.matrix-cell {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.matrix-cell:hover {
    background: rgba(255,255,255,0.04);
}

.matrix-cell.correct {
    border-color: rgba(46,125,50,0.3);
    background: rgba(46,125,50,0.02);
}

.matrix-cell.correct:hover {
    border-color: rgba(46,125,50,0.5);
    background: rgba(46,125,50,0.04);
}

.matrix-cell.incorrect {
    border-color: rgba(198,40,40,0.3);
    background: rgba(198,40,40,0.02);
}

.matrix-cell.incorrect:hover {
    border-color: rgba(198,40,40,0.5);
    background: rgba(198,40,40,0.04);
}

.matrix-value {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: #e2dfd8;
    line-height: 1.1;
}

.matrix-cell.correct .matrix-value {
    color: #81c784;
}

.matrix-cell.incorrect .matrix-value {
    color: #e57373;
}

.matrix-cell-lbl {
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 6px;
    color: rgba(220,210,185,0.4);
}

/* ---- PROFILE GLASSMORPHISM EXT ---- */
.profile-card {
    width: 100%;
    box-sizing: border-box;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(212, 175, 55, 0.14);
    border-radius: 16px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    padding: 22px 20px 18px;
    margin-bottom: 10px;
}

.profile-card:hover {
    border-color: rgba(212, 175, 55, 0.26);
    box-shadow: 0 6px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}

.profile-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 60px;
    background: radial-gradient(ellipse 90% 100% at 50% 0%, rgba(212, 175, 55, 0.09) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.profile-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 50%, #f0d060 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 700;
    color: #0a0800;
    letter-spacing: 0.5px;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.25), 0 0 20px rgba(212,175,55,0.25), 0 3px 12px rgba(0,0,0,0.4);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
    position: relative;
    z-index: 1;
    margin-bottom: 14px;
}

.profile-card:hover .profile-avatar {
    box-shadow: 0 0 0 2px rgba(212,175,55,0.45), 0 0 26px rgba(212,175,55,0.35), 0 3px 14px rgba(0,0,0,0.5);
    transform: scale(1.04);
}

.profile-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 19px;
    font-weight: 600;
    color: #f0e8cc;
    letter-spacing: 0.5px;
    line-height: 1.2;
    margin: 0 0 5px 0;
    position: relative;
    z-index: 1;
}

.profile-role {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212, 175, 55, 0.45);
    font-weight: 500;
    line-height: 1;
    position: relative;
    z-index: 1;
}

.contact-card {
    width: 100%;
    box-sizing: border-box;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(212, 175, 55, 0.14);
    border-radius: 16px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    padding: 8px 12px 12px;
}

.contact-card:hover {
    border-color: rgba(212, 175, 55, 0.22);
}

.profile-link {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    text-decoration: none !important;
    padding: 9px 10px !important;
    border-radius: 9px !important;
    background: transparent !important;
    transition: background 0.2s ease, transform 0.2s ease !important;
    color: inherit !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

.profile-link:hover {
    background: rgba(212, 175, 55, 0.07) !important;
    transform: translateX(2px) !important;
    text-decoration: none !important;
}

.profile-link-icon {
    font-size: 12px;
    color: rgba(212, 175, 55, 0.6);
    flex-shrink: 0;
    width: 14px;
    text-align: center;
    text-decoration: none !important;
    transition: color 0.2s ease;
}

.profile-link:hover .profile-link-icon {
    color: rgba(212, 175, 55, 0.9);
}

.profile-link-text {
    font-size: 12px;
    color: rgba(200, 185, 140, 0.55);
    font-weight: 400;
    letter-spacing: 0.2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    transition: color 0.2s ease;
    flex: 1;
    min-width: 0;
    text-decoration: none !important;
}

.profile-link:hover .profile-link-text {
    color: rgba(212, 175, 55, 0.82);
}

.sidebar-version {
    text-align: center;
    padding: 16px 0 24px;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200, 185, 140, 0.18);
    font-weight: 400;
    transition: color 0.3s ease;
}

.sidebar-version:hover {
    color: rgba(200, 185, 140, 0.3);
}
"""

# STATS_CSS contains styles specific to the Stats page (pages/stats.py)
STATS_CSS = """
/* ---- HERO OVERRIDES ---- */
.hero-wrapper {
    padding: 64px 60px 40px;
}

.hero-title {
    font-size: clamp(52px, 7vw, 88px);
    margin-bottom: 18px;
}

/* ---- SECTION HEADER ---- */
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

/* ---- STAT BADGE ---- */
.stat-badge {
    display: inline-block;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 100px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(212,175,55,0.8);
    letter-spacing: 0.5px;
    margin: 0.2rem;
}

/* ---- WIN BAR ---- */
.win-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 6px;
    width: 100%;
    margin: 4px 0 12px 0;
    overflow: hidden;
}

.win-bar-fill {
    height: 6px;
    border-radius: 100px;
    background: linear-gradient(90deg, #b8962e, #d4af37, #f0d060);
    box-shadow: 0 0 8px rgba(212,175,55,0.3);
}

/* ---- DATAFRAME ---- */
.stDataFrame {
    border: 1px solid rgba(212,175,55,0.12) !important;
    border-radius: 14px !important;
    overflow: hidden;
}
"""

# LIVE_MATCH_CSS contains styles specific to the Live Match page (pages/live_match.py)
LIVE_MATCH_CSS = """
/* ---- HERO OVERRIDES ---- */
.hero-wrapper {
    padding: 48px 60px 32px;
}

.hero-eyebrow {
    margin-bottom: 14px;
}

.hero-title {
    font-size: clamp(36px, 4vw, 56px);
    margin-bottom: 10px;
}

/* ---- LIVE PULSE ---- */
@keyframes live-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50% { opacity: 0.8; box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}

.live-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #ef4444;
    display: inline-block;
    animation: live-pulse 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-size: 11px;
    color: #ef4444;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 8px;
    width: fit-content;
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

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: 64px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
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

/* ---- MATCH CARD ---- */
.match-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.25s ease;
}

.match-card:hover {
    border-color: rgba(212,175,55,0.25);
    background: rgba(212,175,55,0.04);
    transform: translateY(-1px);
}

.match-card .teams {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 600;
    color: #f0e8cc;
    margin-bottom: 6px;
}

.match-card .score-line {
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    color: #d4c080;
    margin-bottom: 4px;
}

.match-card .status {
    font-size: 11px;
    color: rgba(200,185,140,0.45);
    letter-spacing: 0.5px;
}

/* ---- SECTION LABEL ---- */
.section-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.4);
    margin-bottom: 16px;
    font-weight: 500;
}

/* ---- STAT PILL ---- */
.stat-pill {
    display: inline-block;
    background: rgba(212,175,55,0.12);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #d4af37;
    margin: 0.2rem;
}

/* ---- INFO BOX ---- */
.info-box {
    background: rgba(212,175,55,0.03);
    border: 1px solid rgba(212,175,55,0.1);
    border-radius: 16px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.info-box .icon {
    font-size: 24px;
    flex-shrink: 0;
}

.info-box .text {
    font-size: 13px;
    color: rgba(200,185,140,0.6);
    line-height: 1.5;
}
"""

def apply_custom_css(page="main"):
    css_content = BASE_CSS
    if page == "main":
        css_content += DASHBOARD_CSS
    elif page == "stats":
        css_content += STATS_CSS
    elif page == "live_match":
        css_content += LIVE_MATCH_CSS
    
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
