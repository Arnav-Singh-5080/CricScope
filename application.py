import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
import os
import html
import joblib
style_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
with open(style_path, 'r') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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
def load_or_train_model():
    if os.path.exists('pipe.pkl'):
        return joblib.load('pipe.pkl')
    
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    
    df = deliveries.merge(matches, left_on='match_id', right_on='id')
    
    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)
    
    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2].copy()
    
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    balls_bowled = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)
    
    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']
    
    overs_bowled = (df['over'] - 1) + (df['ball'] / 6)
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)
    
    final_df = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr', 'result']]
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
    joblib.dump(pipe, 'pipe.pkl')
    return pipe

pipe = load_or_train_model()

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

    st.markdown('<div style="height:1px; background:rgba(212,175,55,0.08); margin:20px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Built By</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="profile-section">'
        '<div class="profile-card">'
        '<div class="profile-avatar">AS</div>'
        '<div class="profile-name">Arnav Singh</div>'
        '<div class="profile-role">ML &middot; Data &middot; Analytics</div>'
        '</div>'
        '<div class="contact-card">'
        '<a href="mailto:itsarnav.singh80@gmail.com" class="profile-link">'
        '<span class="profile-link-icon">&#9993;</span>'
        '<span class="profile-link-text">itsarnav.singh80@gmail.com</span>'
        '</a>'
        '<a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank" class="profile-link">'
        '<span class="profile-link-icon">in</span>'
        '<span class="profile-link-text">linkedin.com/in/arnav-singh</span>'
        '</a>'
        '<a href="https://github.com/Arnav-Singh-5080" target="_blank" class="profile-link">'
        '<span class="profile-link-icon">&#9670;</span>'
        '<span class="profile-link-text">Arnav-Singh-5080</span>'
        '</a>'
        '</div>'
        '</div>'
        '<div class="sidebar-version">CricScope v2.0 &middot; IPL Edition</div>',
        unsafe_allow_html=True
    )

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
                                box-shadow:0 0 20px {html.escape(str(tdata['color']))}50;
                                display:flex;align-items:center;justify-content:center;">
                        <img src="{tdata['logo']}"
                             style="width:100%;height:100%;object-fit:cover;
                                    mix-blend-mode:screen;border-radius:50%;" />
                    </div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:18px; font-weight:600;
                                color:{html.escape(str(tdata['color']))}; letter-spacing:2px; margin-top:12px;">
                        {html.escape(str(tdata['abbr']))}
                    </div>
                    <div style="font-size:10px; color:rgba(200,185,140,0.35); margin-top:4px;
                                letter-spacing:0.5px;">
                        {html.escape(str(team_name))}
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
        cities = [
            'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 
            'Cape Town', 'Centurion', 'Chandigarh', 'Chennai', 'Cuttack', 
            'Delhi', 'Dharamsala', 'Durban', 'East London', 'Hyderabad', 
            'Indore', 'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 
            'Kochi', 'Kolkata', 'Mohali', 'Mumbai', 'Nagpur', 
            'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 
            'Sharjah', 'Visakhapatnam'
        ]
        selected_city = st.selectbox("Select Host City", cities, index=cities.index('Mumbai'), key="city")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Match State</div>', unsafe_allow_html=True)
        target = st.number_input("Target Score", min_value=50, max_value=300, value=180, step=1)
        score = st.number_input("Current Score", min_value=0, max_value=target - 1, value=50, step=1)
        col_ov, col_bl, col_wk = st.columns(3)
        with col_ov:
            overs = st.number_input("Overs", min_value=0, max_value=20, value=10, step=1)
        with col_bl:
            balls = st.number_input("Balls", min_value=0, max_value=5, value=0, step=1)
        with col_wk:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2, step=1)
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
                    {html.escape(str(t1['abbr']))}
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
                    {html.escape(str(t2['abbr']))}
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
        balls_bowled = (overs * 6) + balls
        balls_left = max(120 - balls_bowled, 0)
        
        overs_bowled = overs + (balls / 6.0)
        crr = score / overs_bowled if overs_bowled > 0 else 0.0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        with st.spinner(""):
            
            # Edge-case handling for final ball/completed innings boundaries
            if runs_left <= 0:
                win = 1.0
                lose = 0.0
            elif balls_left <= 0:
                win = 0.0
                lose = 1.0
            else:
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
                    <div class="prediction-label">Batting Team · {html.escape(str(t1['abbr']))}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
                        {html.escape(str(batting_team))}
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
                    <div class="prediction-label">Bowling Team · {html.escape(str(t2['abbr']))}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
                        {html.escape(str(bowling_team))}
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
                        {html.escape(str(verdict))} favoured to win
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

    st.markdown('</div>', unsafe_allow_html=True)  # close main-pad
