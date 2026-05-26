import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =================================== CONFIG ===================================
st.set_page_config(page_title="CricScope", layout="wide", initial_sidebar_state="expanded")

# =================================== SESSION STATE ===================================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# =================================== LUXURY CSS ===================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { font-family: 'DM Sans', sans-serif; color: #e2dfd8; }

[data-testid="stAppViewContainer"] {
    background: #080808;
    background-image: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
                      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%);
}

#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; display: none; }

.input-card, .prediction-card, .team-vs-wrapper {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px 32px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: 72px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 40%, #e8c84a 100%);
    color: #0a0800;
    height: 52px;
    font-weight: 600;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

# =================================== TEAM DATA ===================================
team_data = {
    "Chennai Super Kings": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg", "abbr": "CSK", "color": "#facc15"},
    "Delhi Capitals": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg", "abbr": "DC", "color": "#3b82f6"},
    "Punjab Kings": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg", "abbr": "PBKS", "color": "#ef4444"},
    "Kolkata Knight Riders": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg", "abbr": "KKR", "color": "#7c3aed"},
    "Mumbai Indians": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg", "abbr": "MI", "color": "#3b82f6"},
    "Rajasthan Royals": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg", "abbr": "RR", "color": "#ec4899"},
    "Royal Challengers Bangalore": {"logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg", "abbr": "RCB", "color": "#dc2626"},
    "Sunrisers Hyderabad": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg", "abbr": "SRH", "color": "#f97316"}
}

# =================================== MODEL ===================================
@st.cache_resource
def train_model():
    try:
        matches = pd.read_csv("matches.csv")
        deliveries = pd.read_csv("deliveries.csv")
    except FileNotFoundError:
        st.error("❌ matches.csv and deliveries.csv not found!")
        st.stop()

    df = deliveries.merge(matches, left_on='match_id', right_on='id')
    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_bowled'] = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - df['balls_bowled']).clip(lower=0)

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = 10 - df.groupby('match_id')['player_dismissed'].cumsum()

    overs_bowled = df['balls_bowled'] / 6.0
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
                   'wickets', 'target', 'crr', 'rrr', 'result']].dropna()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000))])
    pipe.fit(final_df.drop('result', axis=1), final_df['result'])
    return pipe

pipe = train_model()

# =================================== SIDEBAR ===================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><span class="sidebar-logo-text">CRICSCOPE</span></div>', unsafe_allow_html=True)
    if st.button("◈ Dashboard"): st.session_state.page = "Dashboard"
    if st.button("◉ Match Analysis"): st.session_state.page = "Analysis"

# =================================== ANALYSIS PAGE ===================================
if st.session_state.page == "Analysis":
    st.markdown('<div class="hero-wrapper"><div class="hero-title">Match Analysis</div></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())
    cities = ['Mumbai', 'Chennai', 'Kolkata', 'Delhi', 'Hyderabad', 'Bangalore', 'Ahmedabad', 'Jaipur']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        batting_team = st.selectbox("Batting Team", teams)
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
        selected_city = st.selectbox("Host City", cities)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        target = st.number_input("Target Score", 50, 300, 180)
        score = st.number_input("Current Score", 0, target-1, 95)
        c1, c2 = st.columns(2)
        with c1: overs = st.slider("Overs Completed", 0, 19, 12)
        with c2: balls_in_over = st.slider("Balls in Current Over", 0, 5, 3)
        wickets = st.number_input("Wickets Fallen", 0, 9, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    t1 = team_data[batting_team]
    t2 = team_data[bowling_team]

    st.markdown(f"""
        <div class="team-vs-wrapper" style="text-align:center;padding:30px;">
            <h2>{t1['abbr']} <span style="color:#d4af37">VS</span> {t2['abbr']}</h2>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)

                total_balls = overs * 6 + balls_in_over
                balls_left = max(120 - total_balls, 0)
                runs_left = target - score

                crr = score / (total_balls / 6.0) if total_balls > 0 else 0.0
                rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 0.0

                if runs_left <= 0:
                    win = 1.0
                elif balls_left <= 0:
                    win = 0.0
                else:
                    input_df = pd.DataFrame([{
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'city': selected_city,
                        'runs_left': runs_left,
                        'balls_left': balls_left,
                        'wickets': 10 - wickets,
                        'target': target,
                        'crr': crr,
                        'rrr': rrr
                    }])

                    # Final Sanitization
                    input_df = input_df.replace([np.inf, -np.inf], np.nan)
                    input_df = input_df.fillna(0)
                    input_df['runs_left'] = input_df['runs_left'].clip(lower=0)
                    input_df['balls_left'] = input_df['balls_left'].clip(lower=0, upper=120)
                    input_df['wickets'] = input_df['wickets'].clip(lower=0, upper=10)

                    proba = pipe.predict_proba(input_df)[0]
                    win = float(proba[1])

                # =============== WIN PROBABILITY GRAPH ===============
                st.subheader("Win Probability Trend")
                remaining_overs = max(1, balls_left // 6)
                overs_list = list(range(overs, overs + remaining_overs + 1))

                # Simulate realistic probability curve
                base = win
                trend = [base]
                for i in range(1, len(overs_list)):
                    change = np.random.uniform(-0.06, 0.03)
                    next_prob = max(0.05, min(0.98, trend[-1] + change))
                    trend.append(next_prob)

                graph_df = pd.DataFrame({
                    "Over": overs_list,
                    "Win Probability (%)": [p * 100 for p in trend]
                })

                st.line_chart(graph_df.set_index("Over"), use_container_width=True, height=320)

                # Main Prediction Display
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">BATTING TEAM</div>
                            <div class="win-probability">{round(win*100)}%</div>
                            <p>{batting_team}</p>
                        </div>
                    """, unsafe_allow_html=True)

                # History
                st.session_state.predictions.append({
                    "Time": datetime.now().strftime("%H:%M"),
                    "Match": f"{t1['abbr']} vs {t2['abbr']}",
                    "Score": f"{score}/{wickets}",
                    "Win %": round(win*100)
                })
                if len(st.session_state.predictions) > 8:
                    st.session_state.predictions.pop(0)

                if st.session_state.predictions:
                    st.subheader("Recent Predictions")
                    st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

st.caption("CricScope v2.1 • Enhanced with Win Probability Graph")