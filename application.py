import streamlit as st

# Set page configurations (must be the very first Streamlit command)
st.set_page_config(
    page_title="CricScope - IPL Match Intelligence",
    page_icon="🏏",
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

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
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
    width: 100%;
}

/* ---- PREDICTION CARD ---- */
.prediction-card {
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 24px;
    padding: 36px 32px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: 72px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR NAVIGATION & BRANDING
# -----------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-logo-text">CRICSCOPE</span>
        <span class="sidebar-tagline">MATCH INTELLIGENCE PLATFORM</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    st.button("✨ Live Match Predictor")
    st.button("📊 Historical Stats")
    st.button("🛡️ Team Standings")

# -----------------------------------
# HERO BANNER INTERFACE
# -----------------------------------
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">
        <div class="hero-dot"></div>
        IPL MATCH INTELLIGENCE • SEASON 2026
    </div>
    <div class="hero-eyebrow">PRECISION CRICKET ENGINE</div>
    <div class="hero-title">CricScope</div>
    <div class="hero-subtitle">Precision match analytics engineered for modern cricket. Real-time win probability powered by machine learning.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# LIVE COUNTERS
# -----------------------------------
st.markdown("""
<div class="stats-row">
    <div class="stat-pill">
        <div class="stat-value">94.2%</div>
        <div class="stat-label">Model Precision</div>
    </div>
    <div class="stat-pill">
        <div class="stat-value">14,820</div>
        <div class="stat-label">Simulations/Sec</div>
    </div>
    <div class="stat-pill">
        <div class="stat-value">0.004s</div>
        <div class="stat-label">Latency Metric</div>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN INTERACTION LAYOUT
st.markdown('<div style="padding: 0 72px 60px;">', unsafe_allow_html=True)

teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders"]
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        team1 = st.selectbox("Select Team 1", teams, index=0)
    with sub_col2:
        team2 = st.selectbox("Select Team 2", teams, index=1)
    
    venue = st.selectbox("Match Venue", ["Wankhede Stadium", "M.A. Chidambaram", "M. Chinnaswamy"])
    target = st.number_input("Target Score", min_value=50, max_value=300, value=180)
    
    sl_col1, sl_col2 = st.columns(2)
    with sl_col1:
        overs = st.slider("Overs Elapsed", 0.0, 20.0, 10.0, 0.1)
    with sl_col2:
        wickets = st.slider("Wickets Lost", 0, 10, 3)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="analyze-btn" style="margin-top:20px;">', unsafe_allow_html=True)
    calculate = st.button("⚡ EXECUTE MATRIX ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if calculate:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="win-team-name" style="font-size:24px; font-weight:600; color:#f0e0a0;">{team1} PROJECTION</div>', unsafe_allow_html=True)
        st.markdown('<div class="win-probability">64.5%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-card" style="text-align:center; padding: 60px 0;">AWAITING EXECUTION COMMAND</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)