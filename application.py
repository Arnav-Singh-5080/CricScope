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
st.set_page_config(page_title="CricScope", layout="wide", initial_sidebar_state="auto")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# -----------------------------------
# RESPONSIVE LUXURY CSS
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ---- BASE ---- */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2dfd8;
}

[data-testid="stAppViewContainer"] {
    background: #080808;
}

/* FIX: Top Margin to bring content down */
.block-container {
    padding-top: 4rem !important; 
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0c0c0c !important;
}

/* Sidebar Buttons Clean Look */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: rgba(220,210,180,0.6) !important;
    text-align: left !important;
    padding: 15px 25px !important;
    width: 100%;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    color: #d4af37 !important;
    background: rgba(212,175,55,0.05) !important;
}

/* ---- HERO & CARDS ---- */
.hero-wrapper {
    text-align: center;
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(34px, 8vw, 68px);
    color: #f0e8cc;
}

.team-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(212,175,55,0.1);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    text-align: center;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# MODEL LOGIC
# -----------------------------------
@st.cache_resource
def train_model():
    try:
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
        df['wickets'] = 10 - df.groupby('match_id')['player_dismissed'].cumsum()
        df['over'] = df['over'].replace(0, 0.1)
        df['crr'] = df['current_score'] / (df['over'] + df['ball'] / 6)
        df['rrr'] = (df['runs_left'] * 6) / df['balls_left']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)
        final_df = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr', 'result']].dropna()
        X = final_df.drop('result', axis=1)
        y = final_df['result']
        preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']), ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])])
        pipe = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000))])
        pipe.fit(X, y)
        return pipe
    except: return None

pipe = train_model()
teams = ["Chennai Super Kings", "Delhi Capitals", "Punjab Kings", "Kolkata Knight Riders", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Sunrisers Hyderabad"]

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.markdown('<h2 style="color:#d4af37; font-family:serif; padding:20px;">CRICSCOPE</h2>', unsafe_allow_html=True)
    
    if st.button("◈  Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun() # Forces navigation immediately
        
    if st.button("◉  Match Analysis"):
        st.session_state.page = "Analysis"
        st.rerun() # Forces navigation immediately

# -----------------------------------
# PAGES
# -----------------------------------
if st.session_state.page == "Dashboard":
    st.markdown('<div class="hero-wrapper"><h1 class="hero-title">CricScope</h1><p>Intelligence Platform</p></div>', unsafe_allow_html=True)
    st.subheader("Active IPL Teams")
    cols = st.columns(2)
    for i, t in enumerate(teams):
        with cols[i%2]:
            st.markdown(f'<div class="team-box">{t}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="hero-wrapper"><h1 class="hero-title">Match Analysis</h1></div>', unsafe_allow_html=True)
    with st.container():
        batting_team = st.selectbox("Batting Team", teams)
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
        target = st.number_input("Target Score", 50, 300, 180)
        score = st.number_input("Score", 0, target-1, 50)
        overs = st.slider("Overs Done", 0, 19, 10)
        wickets = st.number_input("Wickets", 0, 9, 2)
        
        if st.button("Predict Win %", use_container_width=True):
            if pipe:
                # Basic calculation for demo
                input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':['Mumbai'],'runs_left':[target-score],'balls_left':[120-(overs*6)],'wickets':[10-wickets],'target':[target],'crr':[score/overs if overs>0 else 0],'rrr':[(target-score)*6/(120-overs*6) if 120-overs*6>0 else 0]})
                res = pipe.predict_proba(input_df)
                st.metric(f"{batting_team} Win Chance", f"{round(res[0][1]*100)}%")
            else:
                st.error("Model Error")
