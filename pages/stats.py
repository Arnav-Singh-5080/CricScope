import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from components.styles import apply_custom_css

#  page config 
st.set_page_config(
    page_title="CricScope · Stats",
    page_icon="📊",
    layout="wide",
)

apply_custom_css("stats")


#  data loading 
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries


matches, deliveries = load_data()

#  page header 
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">IPL Historical Intelligence · 2008–2020</div>
    <div class="hero-title" style="font-size:clamp(36px,4vw,56px); margin-bottom:10px;">Stats &amp; Records</div>
    <div class="hero-subtitle">
        Historical team performance · Head-to-head records · Venue analysis
    </div>
</div>
""", unsafe_allow_html=True)

#  team list 
all_teams = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))


# SECTION 1 — Head-to-Head

st.markdown('<div style="padding: 0 60px 60px;">', unsafe_allow_html=True)

st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
            color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
    Head-to-Head Record
</div>
<div class="section-title">⚔️ Team Matchup</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0, key="h2h_a")
with col2:
    remaining = [t for t in all_teams if t != team_a]
    team_b = st.selectbox("Team B", remaining, index=remaining.index("Chennai Super Kings") if "Chennai Super Kings" in remaining else 0, key="h2h_b")
    
#Example value
team_a = "Chennai Super KIngs"
team_b = "Mumbai Indians"

team_a_wins = 11
team_b_wins = 17

#Pie Chart
fig = go.Figure(data=[go.Pie(
    labels=[f"{team_a}Wins",f"{team_b}Wins"],
    values=[11,17],
    hole=0.5
)])

fig.update_layout(
    paper_bgcolor="#0E1117",
    font_color="white",
    margin=dict(l=20,r=20,t=20,b=20))

st.plotly_chart(fig, use_container_width=True)
# filter h2h matches
h2h = matches[
    ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
    ((matches["team1"] == team_b) & (matches["team2"] == team_a))
].copy()

if h2h.empty:
    st.info("No head-to-head matches found between these two teams.")
else:
    total = len(h2h)
    a_wins = (h2h["winner"] == team_a).sum()
    b_wins = (h2h["winner"] == team_b).sum()
    no_result = total - a_wins - b_wins

    a_pct = round(a_wins / total * 100, 1) if total else 0
    b_pct = round(b_wins / total * 100, 1) if total else 0

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:2.2rem; font-weight:500; color:#e8d89a; line-height:1; margin-bottom:8px;">{a_wins}</div>
            <div style="color:rgba(200,185,140,0.45); font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">{team_a} Wins</div>
            <div class="win-bar-wrap" style="margin-top:14px;"><div class="win-bar-fill" style="width:{a_pct}%;"></div></div>
            <span class="stat-badge">{a_pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:2.2rem; font-weight:500; color:#e8d89a; line-height:1; margin-bottom:8px;">{total}</div>
            <div style="color:rgba(200,185,140,0.45); font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">Total Matches</div>
            <div style="margin-top:14px; color:rgba(200,185,140,0.3); font-size:11px; letter-spacing:1px; text-transform:uppercase;">No Result: {no_result}</div>
        </div>
        """, unsafe_allow_html=True)
    with cc:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:2.2rem; font-weight:500; color:#e8d89a; line-height:1; margin-bottom:8px;">{b_wins}</div>
            <div style="color:rgba(200,185,140,0.45); font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">{team_b} Wins</div>
            <div class="win-bar-wrap" style="margin-top:14px;"><div class="win-bar-fill" style="width:{b_pct}%;"></div></div>
            <span class="stat-badge">{b_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    # recent 5 matches
    st.markdown("""
    <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                color:rgba(212,175,55,0.4);margin:24px 0 12px;font-weight:500;">
        Recent Encounters
    </div>
    """, unsafe_allow_html=True)
    recent = h2h[["Season", "date", "venue", "winner", "win_by_runs", "win_by_wickets"]].tail(5).iloc[::-1]
    recent.columns = ["Season", "Date", "Venue", "Winner", "Win by Runs", "Win by Wickets"]
    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
    )

st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(212,175,55,0.15), transparent); margin:40px 0;"></div>', unsafe_allow_html=True)

# SECTION 2 — Top Run Scorers & Wicket Takers per Team 
st.markdown("""
<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
            color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
    Player Performance by Team
</div>
<div class="section-title">🏏 Top Performers</div>
""", unsafe_allow_html=True)

selected_team = st.selectbox("Select Team", all_teams, key="player_team")

# batting — runs scored while batting for selected team
# deliveries has batting_team column
team_batting = deliveries[deliveries["batting_team"] == selected_team]
top_batters = (
    team_batting.groupby("batsman")["batsman_runs"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_batters.columns = ["Player", "Runs"]

# bowling — wickets taken while bowling against selected team's opponent
# wicket_kind excludes run outs (credited to fielder not bowler)
team_bowling = deliveries[deliveries["bowling_team"] == selected_team]
wicket_deliveries = team_bowling[
    team_bowling["dismissal_kind"].notna() &
    (~team_bowling["dismissal_kind"].isin(["run out", "retired hurt", "obstructing the field"]))
]
top_bowlers = (
    wicket_deliveries.groupby("bowler")["dismissal_kind"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_bowlers.columns = ["Player", "Wickets"]

col_bat, col_bowl = st.columns(2)

with col_bat:
    st.markdown(f"""
    <div class="glass-card" style="padding:20px 24px; margin-bottom:16px;">
        <div style="font-size:10px; letter-spacing:2px; text-transform:uppercase;
                    color:rgba(212,175,55,0.5); font-weight:500;">
            Top Run Scorers — {selected_team}
        </div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in top_batters.iterrows():
        bar_pct = int(row["Runs"] / top_batters["Runs"].max() * 100)
        rank_color = "#d4af37" if i == 0 else "rgba(200,185,140,0.35)"
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:0.5rem; gap:0.8rem;">
            <span style="color:{rank_color}; font-family:'DM Mono',monospace; font-size:11px; width:1.4rem;">#{i+1}</span>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; color:#e2dfd8;">{row['Player']}</span>
                    <span class="stat-badge">{int(row['Runs'])} runs</span>
                </div>
                <div class="win-bar-wrap" style="margin:4px 0 0 0;">
                    <div class="win-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_bowl:
    st.markdown(f"""
    <div class="glass-card" style="padding:20px 24px; margin-bottom:16px;">
        <div style="font-size:10px; letter-spacing:2px; text-transform:uppercase;
                    color:rgba(212,175,55,0.5); font-weight:500;">
            Top Wicket Takers — {selected_team}
        </div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in top_bowlers.iterrows():
        bar_pct = int(row["Wickets"] / top_bowlers["Wickets"].max() * 100)
        rank_color = "#d4af37" if i == 0 else "rgba(200,185,140,0.35)"
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:0.5rem; gap:0.8rem;">
            <span style="color:{rank_color}; font-family:'DM Mono',monospace; font-size:11px; width:1.4rem;">#{i+1}</span>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; color:#e2dfd8;">{row['Player']}</span>
                    <span class="stat-badge">{int(row['Wickets'])} wkts</span>
                </div>
                <div class="win-bar-wrap" style="margin:4px 0 0 0;">
                    <div class="win-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(212,175,55,0.15), transparent); margin:40px 0;"></div>', unsafe_allow_html=True)

# SECTION 3 — Venue-wise Team Performance

st.markdown("""
<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
            color:rgba(212,175,55,0.4);margin-bottom:8px;font-weight:500;">
    Venue Performance
</div>
<div class="section-title">🏟️ Ground Analysis</div>
""", unsafe_allow_html=True)

venue_team = st.selectbox("Select Team for Venue Analysis", all_teams, key="venue_team")

# matches where team played (home or away)
team_matches = matches[
    (matches["team1"] == venue_team) | (matches["team2"] == venue_team)
].copy()
team_matches["won"] = (team_matches["winner"] == venue_team).astype(int)

venue_stats = (
    team_matches.groupby("venue")
    .agg(
        Played=("won", "count"),
        Won=("won", "sum"),
    )
    .reset_index()
)
venue_stats["Lost"] = venue_stats["Played"] - venue_stats["Won"]
venue_stats["Win %"] = (venue_stats["Won"] / venue_stats["Played"] * 100).round(1)
venue_stats = venue_stats.sort_values("Played", ascending=False).head(10)

st.markdown(f"""
<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
            color:rgba(212,175,55,0.4);margin:24px 0 16px;font-weight:500;">
    Top venues by matches played — {venue_team}
</div>
""", unsafe_allow_html=True)

for _, row in venue_stats.iterrows():
    win_pct = row["Win %"]
    bar_color = "#b8962e" if win_pct >= 50 else "#8b5e3c"
    st.markdown(f"""
    <div class="glass-card" style="padding:20px 24px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:13px; color:#e2dfd8; letter-spacing:0.3px;">{row['venue']}</span>
            <div>
                <span class="stat-badge">P {int(row['Played'])}</span>
                <span class="stat-badge">W {int(row['Won'])}</span>
                <span class="stat-badge">L {int(row['Lost'])}</span>
                <span class="stat-badge" style="color:#f0d060; border-color:rgba(240,208,96,0.3);">{win_pct}%</span>
            </div>
        </div>
        <div class="win-bar-wrap">
            <div style="height:6px; border-radius:100px; width:{win_pct}%;
                        background:linear-gradient(90deg, {bar_color}, #f0d060);
                        box-shadow:0 0 8px rgba(212,175,55,0.25);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

#  footer 
st.markdown("""
<div style='text-align:center; padding:40px 0 24px; font-size:9px; letter-spacing:1.5px;
            text-transform:uppercase; color:rgba(200,185,140,0.18);'>
    CricScope Stats &middot; IPL 2008–2020 &middot; Data via Kaggle
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-pad