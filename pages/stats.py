import pandas as pd
import streamlit as st

from theme_utils import init_theme, render_theme_toggle, theme_stylesheet

#  page config
st.set_page_config(
    page_title="CricScope · Stats",
    page_icon="📊",
    layout="wide",
)

theme = init_theme()

with st.sidebar:
    st.markdown(
        "<div style='padding:28px 20px 8px;font-family:Cormorant Garamond,serif;"
        "font-size:22px;font-weight:600;letter-spacing:2px;color:#d4af37;'>CRICSCOPE</div>",
        unsafe_allow_html=True,
    )
    render_theme_toggle()

st.markdown(theme_stylesheet(st.session_state.theme), unsafe_allow_html=True)

# shared CSS (dark-gold aesthetic + theme-aware classes)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    color: #e8e0d0;
    font-family: 'DM Sans', sans-serif;
}

.gold { color: #d4af37; }

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #d4af37;
    margin-bottom: 0.4rem;
    letter-spacing: 0.04em;
}

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

.win-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin: 4px 0 12px 0;
    overflow: hidden;
}
.win-bar-fill {
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(90deg, #d4af37, #f0d060);
}

h1, h2, h3 { font-family: 'Cormorant Garamond', serif; color: #d4af37; }

.stats-theme-body .stats-subtitle { color: #888; }
.stats-theme-body .stats-muted { color: #aaa; }
.stats-theme-body .stats-footer { color: #444; }
.stats-theme-body .stats-card-value { color: #d4af37; }
.stats-theme-body .stats-card-value-muted { color: #888; }
.stats-theme-body .stats-venue-text { color: #e8e0d0; }
.stats-theme-body .stats-rank-muted { color: #888; }
.stats-theme-body .stats-player-name { font-size: 0.88rem; color: #e8e0d0; }

.stats-pad {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 clamp(32px, 6vw, 80px);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="stats-theme-body stats-pad">', unsafe_allow_html=True)

#  data loading
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries


matches, deliveries = load_data()

#  page header
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size:3rem; margin-bottom:0;'>📊 CricScope Stats</h1>
    <p class="stats-subtitle" style='font-size:1rem; margin-top:0.3rem;'>
        Historical team performance · Head-to-head records · Venue analysis
    </p>
</div>
""", unsafe_allow_html=True)

#  team list
all_teams = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))


# SECTION 1 — Head-to-Head

st.markdown('<div class="section-title">⚔️ Head-to-Head Record</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0, key="h2h_a")
with col2:
    remaining = [t for t in all_teams if t != team_a]
    team_b = st.selectbox("Team B", remaining, index=remaining.index("Chennai Super Kings") if "Chennai Super Kings" in remaining else 0, key="h2h_b")

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
            <div class="stats-card-value" style="font-size:2.2rem; font-weight:700;">{a_wins}</div>
            <div class="stats-muted" style="font-size:0.85rem;">{team_a} Wins</div>
            <div class="win-bar-wrap"><div class="win-bar-fill" style="width:{a_pct}%;"></div></div>
            <span class="stat-pill">{a_pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="stats-card-value-muted" style="font-size:2.2rem; font-weight:700;">{total}</div>
            <div class="stats-muted" style="font-size:0.85rem;">Total Matches</div>
            <div class="stats-muted" style="margin-top:0.5rem; font-size:0.8rem;">No Result: {no_result}</div>
        </div>
        """, unsafe_allow_html=True)
    with cc:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="stats-card-value" style="font-size:2.2rem; font-weight:700;">{b_wins}</div>
            <div class="stats-muted" style="font-size:0.85rem;">{team_b} Wins</div>
            <div class="win-bar-wrap"><div class="win-bar-fill" style="width:{b_pct}%;"></div></div>
            <span class="stat-pill">{b_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    # recent 5 matches
    st.markdown("**Recent Encounters**")
    recent = h2h[["Season", "date", "venue", "winner", "win_by_runs", "win_by_wickets"]].tail(5).iloc[::-1]
    recent.columns = ["Season", "Date", "Venue", "Winner", "Win by Runs", "Win by Wickets"]
    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# SECTION 2 — Top Run Scorers & Wicket Takers per Team
st.markdown('<div class="section-title">🏏 Player Performance by Team</div>', unsafe_allow_html=True)

selected_team = st.selectbox("Select Team", all_teams, key="player_team")

# batting — runs scored while batting for selected team
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
    <div class="glass-card">
        <div class="gold" style="font-size:1rem; font-weight:600; margin-bottom:0.8rem;">
            🏏 Top Run Scorers — {selected_team}
        </div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in top_batters.iterrows():
        bar_pct = int(row["Runs"] / top_batters["Runs"].max() * 100)
        rank_color = "#d4af37" if i == 0 else ""
        rank_class = "stats-card-value" if i == 0 else "stats-rank-muted"
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:0.5rem; gap:0.8rem;">
            <span class="{rank_class}" style="font-family:'DM Mono'; width:1.2rem;">#{i+1}</span>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between;">
                    <span class="stats-player-name">{row['Player']}</span>
                    <span class="stat-pill">{int(row['Runs'])} runs</span>
                </div>
                <div class="win-bar-wrap" style="margin:3px 0 0 0;">
                    <div class="win-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_bowl:
    st.markdown(f"""
    <div class="glass-card">
        <div class="gold" style="font-size:1rem; font-weight:600; margin-bottom:0.8rem;">
            🎯 Top Wicket Takers — {selected_team}
        </div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in top_bowlers.iterrows():
        bar_pct = int(row["Wickets"] / top_bowlers["Wickets"].max() * 100)
        rank_class = "stats-card-value" if i == 0 else "stats-rank-muted"
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:0.5rem; gap:0.8rem;">
            <span class="{rank_class}" style="font-family:'DM Mono'; width:1.2rem;">#{i+1}</span>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between;">
                    <span class="stats-player-name">{row['Player']}</span>
                    <span class="stat-pill">{int(row['Wickets'])} wkts</span>
                </div>
                <div class="win-bar-wrap" style="margin:3px 0 0 0;">
                    <div class="win-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# SECTION 3 — Venue-wise Team Performance

st.markdown('<div class="section-title">🏟️ Venue Performance</div>', unsafe_allow_html=True)

venue_team = st.selectbox("Select Team for Venue Analysis", all_teams, key="venue_team")

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

st.markdown(f"**Top venues by matches played — {venue_team}**")

for _, row in venue_stats.iterrows():
    win_pct = row["Win %"]
    bar_color = "#d4af37" if win_pct >= 50 else "#8b5e3c"
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem 1.5rem; margin-bottom:0.7rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span class="stats-venue-text" style="font-size:0.9rem;">{row['venue']}</span>
            <div>
                <span class="stat-pill">P {int(row['Played'])}</span>
                <span class="stat-pill">W {int(row['Won'])}</span>
                <span class="stat-pill">L {int(row['Lost'])}</span>
                <span class="stat-pill" style="color:#f0d060;">{win_pct}%</span>
            </div>
        </div>
        <div class="win-bar-wrap">
            <div style="height:10px; border-radius:8px; width:{win_pct}%;
                        background:linear-gradient(90deg, {bar_color}, #f0d060);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

#  footer
st.markdown("""
<div class="stats-theme-body stats-footer" style='text-align:center; padding:2rem 0; font-size:0.8rem;'>
    CricScope Stats · IPL 2008–2020 · Data via Kaggle
</div>
""", unsafe_allow_html=True)
