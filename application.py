import streamlit as st

# Set page configurations (must be the very first Streamlit command)
st.set_page_config(
    page_title="Luxury Match Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# LUXURY CSS WITH FIXED LAYOUT & PRESERVED HEADER
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ---- RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; }

/* Fixed global page wrapper to prevent browser white background bleeding */
.stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2dfd8;
    background-color: #080808 !important;
}

[data-testid="stAppViewContainer"] {
    background: #080808 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%) !important;
    min-height: 100vh;
}

/* ---- FIXED HEADER PROTECTION ---- */
/* Header background matched with page theme to fix the white block while preserving elements */
[data-testid="stHeader"] {
    background-color: #080808 !important;
    background-image: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%) !important;
    z-index: 99;
}

/* Ensure header icons/buttons are visible on the dark background */
[data-testid="stHeader"] button, [data-testid="stHeader"] a {
    color: #e2dfd8 !important;
}

/* Hide only default Streamlit decoration line at the very top, nothing else */
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0c0c0c !important;
    border-right: 1px solid rgba(212,175,55,0.12);
    width: 300px !important;
    min-width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
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

/* ---- MAIN CONTENT AREA SAFE MANAGED PADDING ---- */
/* Set safe margins so content doesn't break under the native header */
.block-container {
    padding-top: 6rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ---- HERO SECTION ---- */
.hero-wrapper {
    padding: 64px 72px 40px;
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
    padding: 24px 72px;
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
    padding: 40px 72px 0;
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
    font-family: 'Cormorant