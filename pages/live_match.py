"""
CricScope · Live Match Tracker
==============================
Auto-refreshing page that fetches live cricket scores from a free API,
parses match state into ML-ready features, runs the prediction pipeline,
and repaints win-probability UI every ~30 seconds with zero manual input.
"""

# pyrefly: ignore [missing-import]
import streamlit as st
from components.styles import apply_custom_css
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import requests
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ───────────────────────────────────────────
#  PAGE CONFIG
# ───────────────────────────────────────────
st.set_page_config(
    page_title="CricScope · Live",
    page_icon="🔴",
    layout="wide",
)
apply_custom_css("live_match")

# ───────────────────────────────────────────
#  AUTO-REFRESH (pure JS, no extra dependency)
# ───────────────────────────────────────────
REFRESH_INTERVAL_MS = 30_000  # 30 seconds

if "live_auto_refresh" not in st.session_state:
    st.session_state.live_auto_refresh = False

# Inject a self-refreshing script when auto-refresh is toggled on
if st.session_state.live_auto_refresh:
    st.markdown(
        f"""
        <script>
            (function() {{
                if (window._cricscope_timer) clearTimeout(window._cricscope_timer);
                window._cricscope_timer = setTimeout(function() {{
                    window.parent.document
                        .querySelectorAll('button[kind="secondary"]')
                        .forEach(function(btn) {{
                            if (btn.innerText.trim() === '↻ Force Refresh') btn.click();
                        }});
                }}, {REFRESH_INTERVAL_MS});
            }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

# Luxury CSS is modularized into components/styles.py

# ───────────────────────────────────────────
#  TEAM MAPPING  (API names → model names)
# ───────────────────────────────────────────
TEAM_NAME_MAP = {
    # CricAPI / common names → names used in training CSVs
    "Chennai Super Kings":       "Chennai Super Kings",
    "CSK":                       "Chennai Super Kings",
    "Delhi Capitals":            "Delhi Capitals",
    "DC":                        "Delhi Capitals",
    "Delhi Daredevils":          "Delhi Capitals",
    "Punjab Kings":              "Punjab Kings",
    "PBKS":                      "Punjab Kings",
    "Kings XI Punjab":           "Punjab Kings",
    "Kolkata Knight Riders":     "Kolkata Knight Riders",
    "KKR":                       "Kolkata Knight Riders",
    "Mumbai Indians":            "Mumbai Indians",
    "MI":                        "Mumbai Indians",
    "Rajasthan Royals":          "Rajasthan Royals",
    "RR":                        "Rajasthan Royals",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "RCB":                       "Royal Challengers Bangalore",
    "Sunrisers Hyderabad":       "Sunrisers Hyderabad",
    "SRH":                       "Sunrisers Hyderabad",
    "Gujarat Titans":            "Gujarat Titans",
    "GT":                        "Gujarat Titans",
    "Lucknow Super Giants":      "Lucknow Super Giants",
    "LSG":                       "Lucknow Super Giants",
}

TEAM_DISPLAY = {
    "Chennai Super Kings":       {"abbr": "CSK",  "color": "#facc15"},
    "Delhi Capitals":            {"abbr": "DC",   "color": "#3b82f6"},
    "Punjab Kings":              {"abbr": "PBKS", "color": "#ef4444"},
    "Kolkata Knight Riders":     {"abbr": "KKR",  "color": "#7c3aed"},
    "Mumbai Indians":            {"abbr": "MI",   "color": "#3b82f6"},
    "Rajasthan Royals":          {"abbr": "RR",   "color": "#ec4899"},
    "Royal Challengers Bangalore": {"abbr": "RCB", "color": "#dc2626"},
    "Sunrisers Hyderabad":       {"abbr": "SRH",  "color": "#f97316"},
    "Gujarat Titans":            {"abbr": "GT",   "color": "#22d3ee"},
    "Lucknow Super Giants":      {"abbr": "LSG",  "color": "#06b6d4"},
}


def resolve_team_name(raw_name: str) -> str:
    """Map an API team name to the canonical model training name."""
    if not raw_name:
        return raw_name
    # Direct lookup
    if raw_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[raw_name]
    # Case-insensitive partial match
    raw_lower = raw_name.lower()
    for key, val in TEAM_NAME_MAP.items():
        if key.lower() in raw_lower or raw_lower in key.lower():
            return val
    return raw_name


# ───────────────────────────────────────────
#  MODEL (same as application.py)
# ───────────────────────────────────────────
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

# ───────────────────────────────────────────
#  LIVE DATA FETCHER
# ───────────────────────────────────────────
CRICKET_API_BASE = "https://api.cricapi.com/v1"

# Free-tier API key — users should replace with their own from cricapi.com
try:
    API_KEY = st.secrets.get("CRICAPI_KEY", "")
except Exception:
    API_KEY = ""


def fetch_live_matches(api_key: str) -> list:
    """Fetch current/recent cricket matches from CricAPI free tier."""
    if not api_key:
        return []
    try:
        resp = requests.get(
            f"{CRICKET_API_BASE}/currentMatches",
            params={"apikey": api_key, "offset": 0},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", [])
    except requests.RequestException:
        pass
    return []


def fetch_match_scorecard(api_key: str, match_id: str) -> dict:
    """Fetch detailed scorecard for a specific match."""
    if not api_key:
        return {}
    try:
        resp = requests.get(
            f"{CRICKET_API_BASE}/match_info",
            params={"apikey": api_key, "id": match_id},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", {})
    except requests.RequestException:
        pass
    return {}


def parse_score_string(score_str: str):
    """
    Parse a score string like '124/3' or '124-3' into (runs, wickets).
    Returns (None, None) if parsing fails.
    """
    if not score_str:
        return None, None
    score_str = score_str.strip()
    for sep in ['/', '-']:
        if sep in score_str:
            parts = score_str.split(sep)
            try:
                runs = int(parts[0].strip())
                wickets = int(parts[1].strip().split()[0])  # handle "3 (18.2 Ov)"
                return runs, wickets
            except (ValueError, IndexError):
                continue
    return None, None


def parse_overs_string(overs_str) -> float:
    """
    Parse overs like '14.2' or '14' into a float.
    Returns 0.0 on failure.
    """
    if overs_str is None:
        return 0.0
    try:
        return float(str(overs_str).strip())
    except ValueError:
        return 0.0


def compute_prediction(batting_team, bowling_team, venue, target,
                       current_score, wickets_fallen, overs_bowled):
    """
    Convert raw match state into ML features and run the prediction pipeline.
    Returns dict with batting_win, bowling_win, crr, rrr or None on failure.
    """
    try:
        # Resolve team names to model-compatible names
        bat_name = resolve_team_name(batting_team)
        bowl_name = resolve_team_name(bowling_team)

        # Convert overs to balls
        completed_overs = int(overs_bowled)
        balls_in_current_over = int(round((overs_bowled - completed_overs) * 10))
        total_balls_bowled = (completed_overs * 6) + balls_in_current_over

        if total_balls_bowled <= 0 or target <= 0:
            return None

        runs_left = target - current_score
        balls_left = 120 - total_balls_bowled

        if balls_left <= 0:
            balls_left = 1  # avoid division by zero at end of innings

        wickets_in_hand = 10 - wickets_fallen
        crr = current_score / (total_balls_bowled / 6) if total_balls_bowled > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Use a generic city if venue is unknown
        city = venue if venue else "Mumbai"

        input_df = pd.DataFrame({
            'batting_team': [bat_name],
            'bowling_team': [bowl_name],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_in_hand],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr],
        })

        proba = pipe.predict_proba(input_df)[0]
        if np.isnan(proba).any():
            return None

        return {
            "batting_win": round(proba[1] * 100),
            "bowling_win": round(proba[0] * 100),
            "crr": round(crr, 2),
            "rrr": round(rrr, 2),
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_in_hand": wickets_in_hand,
        }

    except Exception:
        return None


# ───────────────────────────────────────────
#  PAGE HEADER
# ───────────────────────────────────────────
st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">Real-Time Match Intelligence</div>
        <div class="live-badge">
            <div class="live-dot"></div>
            Live Tracker
        </div>
        <div class="hero-title">Live Match</div>
        <div class="hero-subtitle">
            Auto-refreshing win probability predictions powered by live cricket data feeds.
            Connect your API key and watch predictions update ball-by-ball.
        </div>
    </div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────
#  API KEY INPUT
# ───────────────────────────────────────────
st.markdown('<div style="padding: 32px 60px 0;">', unsafe_allow_html=True)

# Try secrets first, then session state, then prompt
if API_KEY:
    user_api_key = API_KEY
elif "cricapi_key" in st.session_state and st.session_state.cricapi_key:
    user_api_key = st.session_state.cricapi_key
else:
    st.markdown("""
        <div class="info-box">
            <div class="icon">🔑</div>
            <div class="text">
                To fetch live match data, you need a free API key from
                <strong>cricapi.com</strong>.<br>
                Sign up at <a href="https://cricapi.com" target="_blank"
                style="color:#d4af37;">cricapi.com</a>,
                copy your API key, and paste it below.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    user_api_key = st.text_input(
        "CricAPI Key",
        type="password",
        placeholder="Enter your CricAPI key...",
        key="cricapi_key_input",
        help="Get a free key from https://cricapi.com",
    )
    if user_api_key:
        st.session_state.cricapi_key = user_api_key

# ───────────────────────────────────────────
#  CONTROLS ROW
# ───────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

with ctrl_col1:
    auto_on = st.toggle(
        "Auto-Refresh (30s)",
        value=st.session_state.live_auto_refresh,
        key="auto_toggle",
    )
    st.session_state.live_auto_refresh = auto_on

with ctrl_col2:
    manual_refresh = st.button("↻ Force Refresh", key="manual_refresh")

with ctrl_col3:
    if st.session_state.live_auto_refresh:
        st.markdown("""
            <div style="display:flex; align-items:center; height:100%; padding-top:8px;">
                <div class="live-dot"></div>
                <span style="font-size:12px; color:rgba(239,68,68,0.7); letter-spacing:1px;">
                    AUTO-REFRESH ACTIVE · 30s INTERVAL
                </span>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────
#  FETCH & DISPLAY
# ───────────────────────────────────────────
st.markdown('<div style="padding: 24px 60px 60px;">', unsafe_allow_html=True)

if not user_api_key:
    # ── DEMO / MANUAL MODE ──
    st.markdown("""
        <div class="section-label">Demo Mode · Manual Input</div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box" style="margin-bottom:24px;">
            <div class="icon">📡</div>
            <div class="text">
                No API key provided — running in <strong>demo mode</strong>.<br>
                Enter match details manually below to simulate live prediction,
                or provide an API key above to auto-fetch live matches.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Manual input form matching the Analysis page pattern
    all_teams = list(TEAM_DISPLAY.keys())

    demo_col1, demo_col2 = st.columns(2, gap="large")

    with demo_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:rgba(212,175,55,0.5);margin-bottom:12px;font-weight:500;">Teams</div>',
            unsafe_allow_html=True,
        )
        demo_bat = st.selectbox("Batting Team", all_teams, key="demo_bat")
        demo_bowl = st.selectbox(
            "Bowling Team",
            [t for t in all_teams if t != demo_bat],
            key="demo_bowl",
        )
        demo_venue = st.text_input("City / Venue", value="Mumbai", key="demo_venue")
        st.markdown('</div>', unsafe_allow_html=True)

    with demo_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:rgba(212,175,55,0.5);margin-bottom:12px;font-weight:500;">Match State</div>',
            unsafe_allow_html=True,
        )
        demo_target = st.number_input("Target", min_value=50, max_value=300, value=185, key="demo_target")
        demo_score_str = st.text_input("Current Score (e.g. 124/3)", value="124/3", key="demo_score")
        demo_overs = st.number_input("Overs Bowled", min_value=0.1, max_value=20.0, value=14.2,
                                     step=0.1, format="%.1f", key="demo_overs")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Win Probability", key="demo_predict", use_container_width=True):
        demo_runs, demo_wickets = parse_score_string(demo_score_str)
        if demo_runs is not None and demo_wickets is not None:
            result = compute_prediction(
                demo_bat, demo_bowl, demo_venue,
                demo_target, demo_runs, demo_wickets, demo_overs,
            )
            if result:
                st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

                bat_data = TEAM_DISPLAY.get(demo_bat, {"abbr": "BAT", "color": "#d4af37"})
                bowl_data = TEAM_DISPLAY.get(demo_bowl, {"abbr": "BOWL", "color": "#888"})

                pc1, pc2 = st.columns(2, gap="large")
                with pc1:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;
                                        color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
                                Batting · {bat_data['abbr']}</div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                        font-weight:500;color:#c8b870;margin-bottom:12px;">{demo_bat}</div>
                            <div class="win-probability">{result['batting_win']}%</div>
                            <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                                        color:rgba(200,185,140,0.35);margin-bottom:16px;">Win Probability</div>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill" style="width:{result['batting_win']}%;"></div>
                            </div>
                            <div style="display:flex;gap:10px;margin-top:18px;">
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{demo_runs}</div>
                                    <div class="metric-chip-label">Score</div>
                                </div>
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{result['runs_left']}</div>
                                    <div class="metric-chip-label">Needed</div>
                                </div>
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{result['balls_left']}</div>
                                    <div class="metric-chip-label">Balls Left</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with pc2:
                    st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                                    border-radius:24px;padding:36px 32px;position:relative;overflow:hidden;">
                            <div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;
                                        color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
                                Bowling · {bowl_data['abbr']}</div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                        font-weight:500;color:#c8b870;margin-bottom:12px;">{demo_bowl}</div>
                            <div style="font-family:'DM Mono',monospace;font-size:64px;font-weight:500;
                                        color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">
                                {result['bowling_win']}%</div>
                            <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                                        color:rgba(200,185,140,0.35);margin-bottom:16px;">Win Probability</div>
                            <div class="prob-bar-track">
                                <div style="height:100%;border-radius:100px;background:rgba(200,185,140,0.2);
                                            width:{result['bowling_win']}%;transition:width 0.8s ease;"></div>
                            </div>
                            <div style="display:flex;gap:10px;margin-top:18px;">
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{result['crr']}</div>
                                    <div class="metric-chip-label">CRR</div>
                                </div>
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{result['rrr']}</div>
                                    <div class="metric-chip-label">RRR</div>
                                </div>
                                <div class="metric-chip">
                                    <div class="metric-chip-value">{result['wickets_in_hand']}</div>
                                    <div class="metric-chip-label">Wickets</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Verdict
                verdict = demo_bat if result['batting_win'] > 50 else demo_bowl
                conf = max(result['batting_win'], result['bowling_win'])
                conf_label = "High" if conf > 75 else "Moderate" if conf > 55 else "Close"

                st.markdown(f"""
                    <div style="margin-top:16px;background:rgba(212,175,55,0.03);
                                border:1px solid rgba(212,175,55,0.1);border-radius:16px;
                                padding:20px 28px;display:flex;align-items:center;
                                justify-content:space-between;">
                        <div>
                            <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                        color:rgba(212,175,55,0.35);margin-bottom:6px;">Model Verdict</div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                        font-weight:500;color:#f0e8cc;">{verdict} favoured to win</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                        color:rgba(212,175,55,0.35);margin-bottom:6px;">Confidence</div>
                            <div style="font-family:'DM Mono',monospace;font-size:20px;color:#d4af37;">
                                {conf_label} · {conf}%</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not compute prediction. Team names may not match the training data.")
        else:
            st.error("Invalid score format. Use format like '124/3'.")

else:
    # ── LIVE API MODE ──
    st.markdown('<div class="section-label">Live Matches</div>', unsafe_allow_html=True)

    with st.spinner("Fetching live match data..."):
        matches_data = fetch_live_matches(user_api_key)

    if not matches_data:
        st.markdown("""
            <div class="info-box">
                <div class="icon">📭</div>
                <div class="text">
                    No live matches found right now, or the API key is invalid.<br>
                    Make sure you have a valid <strong>CricAPI</strong> key.
                    Live matches appear here when games are in progress.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Filter for cricket matches with score data
        live_matches = []
        for m in matches_data:
            if m.get("score") and len(m.get("score", [])) > 0:
                live_matches.append(m)

        if not live_matches:
            st.markdown("""
                <div class="info-box">
                    <div class="icon">⏳</div>
                    <div class="text">
                        Matches found but no live scoring data available yet.<br>
                        Scores will appear here once innings are in progress.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Let user select a match
            match_options = {}
            for m in live_matches:
                teams = m.get("teams", [])
                label = " vs ".join(teams) if teams else m.get("name", "Unknown Match")
                status = m.get("status", "")
                match_options[f"{label}  —  {status}"] = m

            selected_label = st.selectbox(
                "Select Match",
                list(match_options.keys()),
                key="live_match_select",
            )

            match = match_options[selected_label]

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

            # Extract match data
            teams = match.get("teams", [])
            scores = match.get("score", [])
            status = match.get("status", "")
            venue = match.get("venue", "")
            match_type = match.get("matchType", "")

            # Display match header
            team1_name = teams[0] if len(teams) > 0 else "Team A"
            team2_name = teams[1] if len(teams) > 1 else "Team B"

            t1_display = TEAM_DISPLAY.get(resolve_team_name(team1_name),
                                          {"abbr": team1_name[:3].upper(), "color": "#d4af37"})
            t2_display = TEAM_DISPLAY.get(resolve_team_name(team2_name),
                                          {"abbr": team2_name[:3].upper(), "color": "#888"})

            st.markdown(f"""
                <div class="glass-card" style="text-align:center; padding:32px;">
                    <div class="live-badge" style="margin:0 auto 16px;">
                        <div class="live-dot"></div>
                        {status}
                    </div>
                    <div style="display:flex;align-items:center;justify-content:center;gap:28px;">
                        <div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:32px;
                                        font-weight:600;color:{t1_display['color']};letter-spacing:2px;">
                                {t1_display['abbr']}</div>
                            <div style="font-size:11px;color:rgba(200,185,140,0.4);margin-top:4px;">
                                {team1_name}</div>
                        </div>
                        <div style="font-family:'Cormorant Garamond',serif;font-size:36px;
                                    font-weight:300;color:rgba(212,175,55,0.2);">vs</div>
                        <div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:32px;
                                        font-weight:600;color:{t2_display['color']};letter-spacing:2px;">
                                {t2_display['abbr']}</div>
                            <div style="font-size:11px;color:rgba(200,185,140,0.4);margin-top:4px;">
                                {team2_name}</div>
                        </div>
                    </div>
                    <div style="font-size:11px;color:rgba(200,185,140,0.3);margin-top:12px;
                                letter-spacing:0.5px;">🏟 {venue}</div>
                </div>
            """, unsafe_allow_html=True)

            # Display all innings scores
            st.markdown('<div class="section-label" style="margin-top:8px;">Scorecard</div>',
                        unsafe_allow_html=True)

            score_cols = st.columns(len(scores)) if scores else []
            for idx, sc in enumerate(scores):
                with score_cols[idx]:
                    inning_name = sc.get("inning", f"Innings {idx+1}")
                    inning_score = f"{sc.get('r', 0)}/{sc.get('w', 0)}"
                    inning_overs = sc.get("o", 0)
                    st.markdown(f"""
                        <div class="glass-card" style="text-align:center;">
                            <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                                        color:rgba(200,185,140,0.4);margin-bottom:8px;">{inning_name}</div>
                            <div style="font-family:'DM Mono',monospace;font-size:28px;
                                        color:#e8d89a;font-weight:500;">{inning_score}</div>
                            <div style="font-size:12px;color:rgba(200,185,140,0.35);
                                        margin-top:4px;">({inning_overs} Ov)</div>
                        </div>
                    """, unsafe_allow_html=True)

            # ── PREDICTION (2nd innings only) ──
            # Try to identify the 2nd innings and compute prediction
            if len(scores) >= 2:
                # First innings total = target + 1
                first_innings = scores[0]
                second_innings = scores[1]

                target = first_innings.get("r", 0) + 1
                current_score = second_innings.get("r", 0)
                wickets_fallen = second_innings.get("w", 0)
                overs_bowled = parse_overs_string(second_innings.get("o", 0))

                # Determine batting team of 2nd innings
                second_inning_name = second_innings.get("inning", "")
                # The inning field usually contains the team name
                batting_2nd = None
                for t in teams:
                    if t.lower() in second_inning_name.lower():
                        batting_2nd = t
                        break
                if not batting_2nd:
                    batting_2nd = team2_name

                bowling_2nd = team1_name if batting_2nd == team2_name else team2_name

                # Extract city from venue
                city = venue.split(",")[-1].strip() if venue else "Mumbai"

                result = compute_prediction(
                    batting_2nd, bowling_2nd, city,
                    target, current_score, wickets_fallen, overs_bowled,
                )

                if result:
                    st.markdown('<div class="section-label" style="margin-top:16px;">Win Prediction</div>',
                                unsafe_allow_html=True)

                    bat_display = TEAM_DISPLAY.get(
                        resolve_team_name(batting_2nd),
                        {"abbr": batting_2nd[:3].upper(), "color": "#d4af37"},
                    )
                    bowl_display = TEAM_DISPLAY.get(
                        resolve_team_name(bowling_2nd),
                        {"abbr": bowling_2nd[:3].upper(), "color": "#888"},
                    )

                    pc1, pc2 = st.columns(2, gap="large")

                    with pc1:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;
                                            color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
                                    Chasing · {bat_display['abbr']}</div>
                                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;
                                            font-weight:500;color:#c8b870;margin-bottom:12px;">
                                    {batting_2nd}</div>
                                <div class="win-probability">{result['batting_win']}%</div>
                                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                                            color:rgba(200,185,140,0.35);margin-bottom:16px;">
                                    Win Probability</div>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill" style="width:{result['batting_win']}%;"></div>
                                </div>
                                <div style="display:flex;gap:10px;margin-top:18px;">
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">
                                            {current_score}/{wickets_fallen}</div>
                                        <div class="metric-chip-label">Score</div>
                                    </div>
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">{result['runs_left']}</div>
                                        <div class="metric-chip-label">Needed</div>
                                    </div>
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">{result['balls_left']}</div>
                                        <div class="metric-chip-label">Balls Left</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with pc2:
                        st.markdown(f"""
                            <div style="background:rgba(255,255,255,0.02);
                                        border:1px solid rgba(255,255,255,0.07);
                                        border-radius:24px;padding:36px 32px;">
                                <div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;
                                            color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
                                    Defending · {bowl_display['abbr']}</div>
                                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;
                                            font-weight:500;color:#c8b870;margin-bottom:12px;">
                                    {bowling_2nd}</div>
                                <div style="font-family:'DM Mono',monospace;font-size:64px;font-weight:500;
                                            color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">
                                    {result['bowling_win']}%</div>
                                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                                            color:rgba(200,185,140,0.35);margin-bottom:16px;">
                                    Win Probability</div>
                                <div class="prob-bar-track">
                                    <div style="height:100%;border-radius:100px;
                                                background:rgba(200,185,140,0.2);
                                                width:{result['bowling_win']}%;
                                                transition:width 0.8s ease;"></div>
                                </div>
                                <div style="display:flex;gap:10px;margin-top:18px;">
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">{result['crr']}</div>
                                        <div class="metric-chip-label">CRR</div>
                                    </div>
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">{result['rrr']}</div>
                                        <div class="metric-chip-label">RRR</div>
                                    </div>
                                    <div class="metric-chip">
                                        <div class="metric-chip-value">{result['wickets_in_hand']}</div>
                                        <div class="metric-chip-label">Wickets</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Verdict
                    verdict = batting_2nd if result['batting_win'] > 50 else bowling_2nd
                    conf = max(result['batting_win'], result['bowling_win'])
                    conf_label = "High" if conf > 75 else "Moderate" if conf > 55 else "Close"

                    st.markdown(f"""
                        <div style="margin-top:16px;background:rgba(212,175,55,0.03);
                                    border:1px solid rgba(212,175,55,0.1);border-radius:16px;
                                    padding:20px 28px;display:flex;align-items:center;
                                    justify-content:space-between;">
                            <div>
                                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                            color:rgba(212,175,55,0.35);margin-bottom:6px;">
                                    Model Verdict</div>
                                <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                            font-weight:500;color:#f0e8cc;">
                                    {verdict} favoured to win</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                            color:rgba(212,175,55,0.35);margin-bottom:6px;">
                                    Confidence</div>
                                <div style="font-family:'DM Mono',monospace;font-size:20px;color:#d4af37;">
                                    {conf_label} · {conf}%</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            elif len(scores) == 1:
                st.markdown("""
                    <div class="info-box" style="margin-top:16px;">
                        <div class="icon">⏳</div>
                        <div class="text">
                            First innings in progress. Win prediction will be available
                            once the second innings begins and a target is set.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Last updated timestamp
    st.markdown(f"""
        <div style="text-align:center;margin-top:32px;padding-bottom:24px;font-size:9px;
                    letter-spacing:1.5px;text-transform:uppercase;color:rgba(200,185,140,0.18);">
            Last fetched · {time.strftime("%H:%M:%S")} ·
            {"Auto-refresh ON" if st.session_state.live_auto_refresh else "Auto-refresh OFF"}
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main padding

# ── Auto-refresh via st.rerun ──
if st.session_state.live_auto_refresh and user_api_key:
    time.sleep(REFRESH_INTERVAL_MS / 1000)
    st.rerun()
