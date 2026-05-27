import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="CricScope Live", page_icon="🏏", layout="wide")

# copying the same theme from application.py so it looks consistent
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

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
section[data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid rgba(212,175,55,0.12);
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #d4af37;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 52px;
    font-weight: 300;
    color: #f0ece4;
    line-height: 1.1;
    margin-bottom: 8px;
}
.page-subtitle {
    font-size: 14px;
    color: #8a8478;
    margin-bottom: 32px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.25);
    color: #d4af37;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 28px;
    text-transform: uppercase;
}
.live-dot {
    width: 7px; height: 7px;
    background: #d4af37;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.scoreboard {
    background: linear-gradient(135deg, #111111 0%, #0f0f0f 100%);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.score-main {
    font-family: 'Cormorant Garamond', serif;
    font-size: 64px;
    font-weight: 600;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 6px 0;
}
.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(200,185,140,0.45);
    text-transform: uppercase;
}
.team-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.win-bar-bg {
    background: #1a1a1a;
    border-radius: 3px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0 4px;
}
.win-bar-fill {
    height: 8px;
    border-radius: 3px;
    background: linear-gradient(90deg, #a07820, #d4af37, #f0d060);
}
.ball-log {
    background: #0d0d0d;
    border: 1px solid rgba(212,175,55,0.08);
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    max-height: 300px;
    overflow-y: auto;
}
.ball-entry { padding: 5px 0; border-bottom: 1px solid #161616; color: rgba(226,223,216,0.55); }
.ball-entry.boundary { color: #f0d060; }
.ball-entry.wicket { color: #e05555; }
.ball-entry.dot { color: #3a3a3a; }

[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid rgba(212,175,55,0.1);
    border-radius: 6px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: rgba(200,185,140,0.45) !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 28px !important;
    color: #f0d060 !important;
}
</style>
""", unsafe_allow_html=True)


# load CSVs and model
@st.cache_data(show_spinner=False)
def load_data():
    try:
        matches = pd.read_csv("matches.csv")
        deliveries = pd.read_csv("deliveries.csv")
        # normalize column names — dataset has inconsistent casing
        matches.columns = matches.columns.str.strip().str.lower()
        deliveries.columns = deliveries.columns.str.strip().str.lower()
        return matches, deliveries
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("pipe.pkl")
    except Exception:
        return None


matches, deliveries = load_data()
pipe = load_model()


# session state
for key, val in {
    "lm_match_id": None,
    "lm_ball_index": 0,
    "lm_running": False,
    "lm_speed": 2,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_data(show_spinner=False)
def get_match_balls(match_id):
    # get only second innings for this match
    data = deliveries[
        (deliveries["match_id"] == match_id) &
        (deliveries["inning"] == 2)
    ].copy().reset_index(drop=True)

    if len(data) == 0:
        return data

    # cumulative score ball by ball
    data["cum_score"] = data["total_runs"].cumsum()

    # handle different column names across dataset versions
    if "is_wicket" in data.columns:
        wkts = data["is_wicket"].fillna(0).astype(int)
    elif "player_dismissed" in data.columns:
        wkts = (
            data["player_dismissed"].fillna("").astype(str).str.strip()
            .apply(lambda x: int(x not in ["", "0", "None", "nan", "NaN"]))
        )
    else:
        wkts = pd.Series([0] * len(data))

    data["cum_wickets"] = wkts.cumsum()
    return data


def get_target(match_id):
    inn1 = deliveries[
        (deliveries["match_id"] == match_id) &
        (deliveries["inning"] == 1)
    ]
    return int(inn1["total_runs"].sum()) + 1


def get_win_prob(batting, bowling, city, target, score, wickets, balls_bowled):
    if pipe is None:
        return None

    runs_left = target - score
    balls_left = 120 - balls_bowled
    wkts_left = 10 - wickets

    if runs_left <= 0:
        return (1.0, 0.0)
    if balls_left <= 0:
        return (0.0, 1.0)

    # crr = runs per over
    overs = balls_bowled / 6
    crr = score / max(overs, 0.1)
    rrr = (runs_left * 6) / max(balls_left, 1)

    try:
        df = pd.DataFrame({
            "batting_team": [batting],
            "bowling_team": [bowling],
            "city": [city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets": [wkts_left],
            "target": [target],
            "crr": [crr],
            "rrr": [rrr],
        })
        p = pipe.predict_proba(df)[0]
        return float(p[1]), float(p[0])
    except Exception:
        return None


def format_ball(row):
    runs = int(row.get("batsman_runs", 0))
    total = int(row.get("total_runs", 0))
    over = int(row.get("over", 0))
    ball = int(row.get("ball", 0))
    prefix = f"Over {over}.{ball}  —  "

    is_w = False
    if "is_wicket" in row:
        is_w = int(row.get("is_wicket", 0)) == 1
    elif "player_dismissed" in row:
        val = str(row.get("player_dismissed", "")).strip()
        is_w = val not in ["", "0", "None", "nan", "NaN"]

    if is_w:
        return prefix + "WICKET!", "wicket"
    if runs == 6:
        return prefix + "SIX! 🏏", "boundary"
    if runs == 4:
        return prefix + "FOUR! 🏏", "boundary"
    if total == 0:
        return prefix + "Dot ball", "dot"
    return prefix + f"{total} run{'s' if total > 1 else ''}", "normal"


# page header
st.markdown('<p class="section-label">IPL Match Intelligence · Season 2025</p>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Live Match Tracker</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Ball-by-ball simulation · Real-time win probability · ML powered</p>', unsafe_allow_html=True)
st.markdown('<div class="live-badge"><span class="live-dot"></span>Live Simulation Active</div>', unsafe_allow_html=True)

# match selector
valid = matches.dropna(subset=["winner", "city", "team1", "team2"])

options = {
    row["id"]: f"{row['season']} | {row['team1']} vs {row['team2']} ({row['city']})"
    for _, row in valid.iterrows()
}

with st.form("match_form"):
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        selected_label = st.selectbox("Select Match", list(options.values()), label_visibility="collapsed")
    with col2:
        speed = st.selectbox("Speed (s)", [1, 2, 3, 5], index=1, label_visibility="collapsed")
    with col3:
        submitted = st.form_submit_button("Lock Match", use_container_width=True)

    if submitted:
        chosen_id = [k for k, v in options.items() if v == selected_label][0]
        st.session_state.lm_match_id = chosen_id
        st.session_state.lm_ball_index = 0
        st.session_state.lm_running = False
        st.session_state.lm_speed = speed
        st.rerun()

if st.session_state.lm_match_id is None:
    st.session_state.lm_match_id = valid.iloc[0]["id"]

# load selected match
mid = st.session_state.lm_match_id
# clear cache so switching matches loads fresh data
get_match_balls.clear()
match_info = valid[valid["id"] == mid].iloc[0]
inn2 = get_match_balls(mid)

if len(inn2) == 0:
    st.error("No second innings data for this match.")
    st.stop()

target = get_target(mid)
batting_team = inn2["batting_team"].iloc[0]
bowling_team = inn2["bowling_team"].iloc[0]
city = str(match_info["city"])

# controls
st.markdown('<p class="section-label" style="margin-top:16px;">Controls</p>', unsafe_allow_html=True)
b1, b2 = st.columns(2)

with b1:
    btn_label = "Pause" if st.session_state.lm_running else "Start Simulation"
    if st.button(btn_label, use_container_width=True):
        st.session_state.lm_running = not st.session_state.lm_running
        st.rerun()

with b2:
    if st.button("Reset", use_container_width=True):
        st.session_state.lm_ball_index = 0
        st.session_state.lm_running = False
        st.rerun()

# advance one ball per rerun when running
idx = st.session_state.lm_ball_index
if st.session_state.lm_running and idx < len(inn2):
    st.session_state.lm_ball_index += 1
    idx = st.session_state.lm_ball_index

current = inn2.iloc[:idx]

if len(current) > 0:
    score = int(current.iloc[-1]["cum_score"])
    wickets = int(current.iloc[-1]["cum_wickets"])
else:
    score = 0
    wickets = 0

balls_bowled = len(current)
overs_done = balls_bowled // 6
balls_rem = balls_bowled % 6
overs_str = f"{overs_done}.{balls_rem}"
runs_left = max(target - score, 0)
balls_left = max(120 - balls_bowled, 0)

# crr = runs scored per over
crr = round(score / max(balls_bowled / 6, 0.1), 2)
rrr = round((runs_left * 6) / max(balls_left, 1), 2)

win_prob = get_win_prob(batting_team, bowling_team, city, target, score, wickets, balls_bowled) if balls_bowled > 0 else None

# scoreboard
st.markdown('<p class="section-label" style="margin-top:24px;">Scoreboard</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="scoreboard">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div class="score-label">Batting</div>
            <div class="team-name">{batting_team}</div>
            <div class="score-main">{score}/{wickets}</div>
            <div class="score-label">Overs {overs_str} &nbsp;·&nbsp; Target {target}</div>
        </div>
        <div style="text-align:right;">
            <div class="score-label">Bowling</div>
            <div class="team-name">{bowling_team}</div>
            <div style="margin-top:12px; font-family:'DM Mono',monospace; font-size:11px; color:rgba(200,185,140,0.4);">
                Needs {runs_left} off {balls_left} balls
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runs Left", runs_left)
m2.metric("Balls Left", balls_left)
m3.metric("CRR", crr)
m4.metric("RRR", rrr)

# win probability bar
if win_prob:
    bat_pct = int(win_prob[0] * 100)
    bowl_pct = 100 - bat_pct
    st.markdown('<p class="section-label" style="margin-top:24px;">Win Probability</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-family:'DM Mono',monospace; font-size:11px; color:#d4af37; letter-spacing:1px;">{batting_team} &nbsp; {bat_pct}%</span>
            <span style="font-family:'DM Mono',monospace; font-size:11px; color:rgba(200,185,140,0.35); letter-spacing:1px;">{bowl_pct}% &nbsp; {bowling_team}</span>
        </div>
        <div class="win-bar-bg">
            <div class="win-bar-fill" style="width:{bat_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ball by ball log
st.markdown('<p class="section-label" style="margin-top:8px;">Ball Log</p>', unsafe_allow_html=True)

if len(current) > 0:
    html = '<div class="ball-log">'
    for _, row in current.tail(150).iloc[::-1].iterrows():
        text, css = format_ball(row)
        html += f'<div class="ball-entry {css}">{text}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="ball-log" style="text-align:center; color:#333; padding:24px;">Press Start Simulation to begin</div>',
        unsafe_allow_html=True
    )

# match result
if idx >= len(inn2):
    st.session_state.lm_running = False
    st.success(f"🏆 {match_info['winner']} won the match!")

# the actual fix for issue #188
if st.session_state.lm_running and idx < len(inn2):
    time.sleep(st.session_state.lm_speed)
    st.rerun()



