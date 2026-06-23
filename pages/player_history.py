import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Player History",
    page_icon="🏏",
    layout="wide"
)

@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    matches.rename(
        columns={"season": "Season"},
        inplace=True
    )

    deliveries.rename(
        columns={"batter": "batsman"},
        inplace=True
    )

    return matches, deliveries
matches, deliveries = load_data()
TEAM_RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants"
}

# Add season information to deliveries
data = deliveries.merge(
    matches[["id", "Season", "venue"]],
    left_on="match_id",
    right_on="id",
    how="left"
)
data["batting_team"] = data["batting_team"].replace(TEAM_RENAME)
data["bowling_team"] = data["bowling_team"].replace(TEAM_RENAME)
players = sorted(data["batsman"].dropna().unique())

st.title("🏏 IPL Player History Dashboard")

selected_player = st.selectbox(
    "🔍 Search Player",
    sorted(players)
)
valid_wickets = [
    "bowled",
    "caught",
    "lbw",
    "stumped",
    "caught and bowled",
    "hit wicket"
]
st.caption(
    "Search using IPL dataset names (V Kohli, RG Sharma, MS Dhoni, AB de Villiers)"
)

player_data = data[data["batsman"] == selected_player]
bowling_data = data[data["bowler"] == selected_player]
# Career Highlights

best_batting_season = "N/A"
best_batting_runs = 0

season_runs_summary = (
    player_data.groupby("Season")["batsman_runs"]
    .sum()
    .reset_index()
)

if not season_runs_summary.empty:
    best_batting_row = season_runs_summary.loc[
        season_runs_summary["batsman_runs"].idxmax()
    ]

    best_batting_season = best_batting_row["Season"]
    best_batting_runs = int(best_batting_row["batsman_runs"])


best_bowling_season = "N/A"
best_bowling_wickets = 0

bowler_wickets_summary = bowling_data[
    bowling_data["dismissal_kind"].isin(valid_wickets)
]

season_wickets_summary = (
    bowler_wickets_summary
    .groupby("Season")
    .size()
    .reset_index(name="Wickets")
)

if not season_wickets_summary.empty:
    best_bowling_row = season_wickets_summary.loc[
        season_wickets_summary["Wickets"].idxmax()
    ]

    best_bowling_season = best_bowling_row["Season"]
    best_bowling_wickets = int(best_bowling_row["Wickets"])
    st.subheader("🏆 Career Summary")

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Best Batting Season",
    best_batting_season
)

s2.metric(
    "Runs",
    best_batting_runs
)

s3.metric(
    "Best Bowling Season",
    best_bowling_season
)

s4.metric(
    "Wickets",
    best_bowling_wickets
)

st.divider()

# Career Stats
total_runs = player_data["batsman_runs"].sum()

balls_faced = len(player_data)

strike_rate = round(
    (total_runs / balls_faced) * 100,
    2
) if balls_faced > 0 else 0
dismissals = player_data[
    player_data["player_dismissed"] == selected_player
].shape[0]

batting_average = (
    round(total_runs / dismissals, 2)
    if dismissals > 0 else total_runs
)
matches_played = player_data["match_id"].nunique()

match_scores = (
    player_data.groupby("match_id")["batsman_runs"]
    .sum()
)

best_match_id = match_scores.idxmax()
highest_score = match_scores.max()
match_scores = (
    player_data.groupby("match_id")["batsman_runs"]
    .sum()
)

fifties = len(
    match_scores[
        (match_scores >= 50) &
        (match_scores < 100)
    ]
)

hundreds = len(
    match_scores[
        match_scores >= 100
    ]
)
total_fours = len(
    player_data[
        player_data["batsman_runs"] == 4
    ]
)

total_sixes = len(
    player_data[
        player_data["batsman_runs"] == 6
    ]
)
boundary_runs = (
    total_fours * 4
) + (
    total_sixes * 6
)

boundary_percentage = (
    (boundary_runs / total_runs) * 100
    if total_runs > 0 else 0
)
matches_played = player_data["match_id"].nunique()

highest_score = (
    player_data.groupby("match_id")["batsman_runs"]
    .sum()
    .max()
)
best_match_id = (
    player_data.groupby("match_id")["batsman_runs"]
    .sum()
    .idxmax()
)

best_match_info = matches[
    matches["id"] == best_match_id
].iloc[0]

best_opponent_team = (
    best_match_info["team2"]
    if best_match_info["team1"] in player_data["batting_team"].unique()
    else best_match_info["team1"]
)

best_match_season = best_match_info["Season"]
match_scores = (
    player_data.groupby("match_id")["batsman_runs"]
    .sum()
)

fifties = len(
    match_scores[
        (match_scores >= 50) &
        (match_scores < 100)
    ]
)

hundreds = len(
    match_scores[
        match_scores >= 100
    ]
)

# ---------------- BOWLING STATS ----------------

bowling_data = data[data["bowler"] == selected_player]

total_balls_bowled = len(bowling_data)

total_runs_conceded = bowling_data["total_runs"].sum()

total_wickets = bowling_data[
    bowling_data["dismissal_kind"].isin(valid_wickets)
].shape[0]
bowler_wickets = bowling_data[
    bowling_data["dismissal_kind"].isin(valid_wickets)
]
economy = (
    total_runs_conceded /
    (total_balls_bowled / 6)
    if total_balls_bowled > 0 else 0
)

bowling_strike_rate = (
    total_balls_bowled /
    total_wickets
    if total_wickets > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Matches", matches_played)
col2.metric("Runs", int(total_runs))
col3.metric("Average", batting_average)
col4.metric("Strike Rate", strike_rate)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Highest Score",
    int(highest_score)
)
col5.metric("50s", fifties)
col6.metric("100s", hundreds)
st.caption(
    f"🔥 Highest Score: {int(highest_score)} vs {best_opponent_team} ({best_match_season})"
)
col7, col8, col9, col10 = st.columns(4)

col7.metric(
    "4️⃣ Fours",
    total_fours
)

col8.metric(
    "6️⃣ Sixes",
    total_sixes
)

col9.metric(
    "🏏 Boundary Runs",
    boundary_runs
)

col10.metric(
    "📊 Boundary %",
    f"{boundary_percentage:.1f}%"
)


st.divider()

st.subheader("🎯 Bowling Statistics")

b1, b2, b3, b4 = st.columns(4)

b1.metric(
    "Wickets",
    int(total_wickets)
)

b2.metric(
    "Balls Bowled",
    int(total_balls_bowled)
)

b3.metric(
    "Economy",
    round(economy, 2)
)

b4.metric(
    "Bowling Strike Rate",
    round(bowling_strike_rate, 2)
)
st.subheader("🏅 Career Milestones")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "1000 Runs",
    "✅" if total_runs >= 1000 else "❌"
)

m2.metric(
    "2000 Runs",
    "✅" if total_runs >= 2000 else "❌"
)

m3.metric(
    "5000 Runs",
    "✅" if total_runs >= 5000 else "❌"
)

m4.metric(
    "100 Wickets",
    "✅" if total_wickets >= 100 else "❌"
)

m5.metric(
    "150 Wickets",
    "✅" if total_wickets >= 150 else "❌"
)

st.divider()
st.subheader("🎯 Wickets by Season")

season_wickets = (
    bowler_wickets
    .groupby("Season")
    .size()
    .reset_index(name="Wickets")
)
season_wickets = season_wickets.sort_values(
    "Season",
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)
if not season_wickets.empty:

    best_bowling = season_wickets.loc[
        season_wickets["Wickets"].idxmax()
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "🏆 Best Bowling Season",
        best_bowling["Season"]
    )

    c2.metric(
        "🎯 Wickets",
        int(best_bowling["Wickets"])
    )
if not season_wickets.empty:
    fig_bowl = px.bar(
        season_wickets,
        x="Season",
        y="Wickets",
        title=f"{selected_player} - Wickets by Season"
    )

    st.plotly_chart(
        fig_bowl,
        use_container_width=True
    )
else:
    st.info("No bowling records available for this player.")
    
st.divider()

st.subheader("🎯 Top Teams Dismissed")

bowling_wickets = bowling_data[
    bowling_data["dismissal_kind"].isin(valid_wickets)
].copy()


team_wickets = (
    bowling_wickets.groupby("batting_team")
    .size()
    .reset_index(name="Wickets")
    .sort_values("Wickets", ascending=False)
)
if not team_wickets.empty:

  if not team_wickets.empty:

    best_team = team_wickets.iloc[0]

    c1, c2 = st.columns(2)

    c1.metric(
        "🎯 Most Dismissed Team",
        best_team["batting_team"]
    )

    c2.metric(
        "🏏 Wickets Taken",
        int(best_team["Wickets"])
    )

else:
    st.info("No wickets available against any team.")
st.dataframe(
    team_wickets.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

fig_team_wickets = px.bar(
    team_wickets.head(10),
    x="batting_team",
    y="Wickets",
    color="Wickets",
    title=f"{selected_player} - Wickets Against Teams"
)

fig_team_wickets.update_layout(
    xaxis_title="Opponent Team",
    yaxis_title="Wickets",
    xaxis_tickangle=-35
)

st.plotly_chart(
    fig_team_wickets,
    use_container_width=True
)
st.divider()

# Runs by season
season_runs = (
    player_data.groupby("Season")["batsman_runs"]
    .sum()
    .reset_index()
)

season_runs = season_runs.sort_values(
    "Season",
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)

if not season_runs.empty:

    best_season = season_runs.loc[
        season_runs["batsman_runs"].idxmax()
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "🏆 Best Season",
        best_season["Season"]
    )

    c2.metric(
        "🏏 Runs in Best Season",
        int(best_season["batsman_runs"])
    )

    fig1 = px.line(
        season_runs,
        x="Season",
        y="batsman_runs",
        markers=True,
        title=f"{selected_player} - Runs by Season"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

else:
    st.info("No batting records available for this player.")
# Strike rate by season
season_sr = (
    player_data.groupby("Season")
    .agg(
        Runs=("batsman_runs", "sum"),
        Balls=("ball", "count")
    )
    .reset_index()
)
season_sr = season_sr.sort_values(
    "Season",
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)
season_sr["Strike Rate"] = (
    season_sr["Runs"] /
    season_sr["Balls"]
) * 100

fig2 = px.bar(
    season_sr,
    x="Season",
    y="Strike Rate",
    title=f"{selected_player} - Strike Rate by Season"
)

st.plotly_chart(fig2, use_container_width=True)
st.divider()

st.subheader("📈 Season-wise Performance Summary")

season_summary = (
    player_data.groupby("Season")
    .agg(
        Runs=("batsman_runs", "sum"),
        Balls=("ball", "count")
    )
    .reset_index()
)

season_summary["Strike Rate"] = (
    season_summary["Runs"] /
    season_summary["Balls"]
) * 100

season_summary["Strike Rate"] = (
    season_summary["Strike Rate"]
    .round(2)
)

season_summary = season_summary.sort_values(
    "Season",
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)

st.dataframe(
    season_summary.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
st.divider()

st.subheader("🏏 Team History")
st.caption(
    f"Teams represented by {selected_player} across IPL seasons"
)
team_history = (
    player_data.groupby(["Season", "batting_team"])
    .size()
    .reset_index(name="Balls")
)

team_history = (
    team_history.sort_values("Balls", ascending=False)
    .drop_duplicates("Season")
    [["Season", "batting_team"]]
)

team_history.columns = ["Season", "Team"]
team_history = team_history.sort_values(
    "Season",
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)
st.dataframe(
    team_history.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
st.divider()

st.subheader("⚔️ Opponent-wise Performance")
st.caption(
    f"Batting record of {selected_player} against different IPL teams"
)
opponent_data = player_data.copy()

opponent_data["Opponent"] = opponent_data.apply(
    lambda row: row["bowling_team"],
    axis=1
)

opponent_stats = (
    opponent_data.groupby("Opponent")
    .agg(
        Runs=("batsman_runs", "sum"),
        Balls=("ball", "count")
    )
    .reset_index()
)

opponent_stats["Strike Rate"] = (
    opponent_stats["Runs"] /
    opponent_stats["Balls"]
) * 100

opponent_stats["Strike Rate"] = opponent_stats["Strike Rate"].round(2)

opponent_stats = opponent_stats.sort_values(
    "Runs",
    ascending=False
)
if not opponent_stats.empty:

    best_opponent = opponent_stats.iloc[0]

    worst_opponent = opponent_stats.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🔥 Best Opponent",
        best_opponent["Opponent"]
    )

    c2.metric(
        "🏏 Runs",
        int(best_opponent["Runs"])
    )

    c3.metric(
        "❄️ Toughest Opponent",
        worst_opponent["Opponent"]
    )

    c4.metric(
        "🏏 Runs",
        int(worst_opponent["Runs"])
    )

else:
    st.info("No opponent statistics available.")

st.dataframe(
    opponent_stats.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
st.divider()
st.subheader("📊 Top Opponents by Runs")

top_opponents = opponent_stats.head(10)

fig3 = px.bar(
    
    top_opponents,
    x="Opponent",
    y="Runs",
    title=f"{selected_player} - Runs Against Opponents"
)
fig3.update_layout(
    xaxis_tickangle=-35
)
st.plotly_chart(fig3, use_container_width=True)
st.divider()

st.subheader("🏟️ Venue Analysis")

venue_stats = (
    player_data.groupby("venue")["batsman_runs"]
    .sum()
    .reset_index()
)

venue_stats = venue_stats.sort_values(
    "batsman_runs",
    ascending=False
)

st.divider()

st.subheader("🏟️ Venue Analysis")

venue_stats = (
    player_data.groupby("venue")["batsman_runs"]
    .sum()
    .reset_index()
)

venue_stats = venue_stats.sort_values(
    "batsman_runs",
    ascending=False
)

if not venue_stats.empty:

    best_venue = venue_stats.iloc[0]

    c1, c2 = st.columns(2)

    c1.metric(
        "🏟️ Favorite Ground",
        best_venue["venue"]
    )

    c2.metric(
        "🏏 Runs Scored",
        int(best_venue["batsman_runs"])
    )

    st.dataframe(
        venue_stats.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    top_venues = venue_stats.head(10)

    fig_venue = px.bar(
        top_venues,
        x="venue",
        y="batsman_runs",
        title=f"{selected_player} - Runs by Venue"
    )

    fig_venue.update_layout(
        xaxis_title="Venue",
        yaxis_title="Runs",
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig_venue,
        use_container_width=True
    )

else:
    st.info("No venue statistics available.")

st.divider()

st.subheader("🎯 Bowling Venue Analysis")

bowling_venue = bowling_data[
    bowling_data["dismissal_kind"].isin(valid_wickets)
]

venue_wickets = (
    bowling_venue.groupby("venue")
    .size()
    .reset_index(name="Wickets")
)

venue_wickets = venue_wickets.sort_values(
    "Wickets",
    ascending=False
)

if not venue_wickets.empty:

    best_venue = venue_wickets.iloc[0]

    c1, c2 = st.columns(2)

    c1.metric(
        "🏟️ Best Bowling Venue",
        best_venue["venue"]
    )

    c2.metric(
        "🎯 Wickets",
        int(best_venue["Wickets"])
    )

else:
    st.info("No bowling venue statistics available.")

st.dataframe(
    venue_wickets.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

top_bowling_venues = venue_wickets.head(10)

fig_bowling_venue = px.bar(
    top_bowling_venues,
    x="venue",
    y="Wickets",
    title=f"{selected_player} - Wickets by Venue"
)

fig_bowling_venue.update_layout(
    xaxis_tickangle=-35
)

st.plotly_chart(
    fig_bowling_venue,
    use_container_width=True
)
st.divider()

st.subheader("🧤 Fielding Statistics")

fielding_data = deliveries[
    deliveries["fielder"] == selected_player
]

total_dismissals = len(fielding_data)

caught_count = len(
    fielding_data[
        fielding_data["dismissal_kind"] == "caught"
    ]
)

runout_count = len(
    fielding_data[
        fielding_data["dismissal_kind"] == "run out"
    ]
)

stumping_count = len(
    fielding_data[
        fielding_data["dismissal_kind"] == "stumped"
    ]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Dismissals",
    total_dismissals
)

col2.metric(
    "Catches",
    caught_count
)

col3.metric(
    "Run Outs",
    runout_count
)

col4.metric(
    "Stumpings",
    stumping_count
)
fielding_summary = pd.DataFrame({
    "Type": ["Caught", "Run Out", "Stumped"],
    "Count": [caught_count, runout_count, stumping_count]
})

st.dataframe(
    fielding_summary,
    use_container_width=True,
    hide_index=True
)

dismissal_breakdown = (
    fielding_data.groupby("dismissal_kind")
    .size()
    .reset_index(name="Count")
)

if not dismissal_breakdown.empty:
    fig4 = px.pie(
        dismissal_breakdown,
        names="dismissal_kind",
        values="Count",
        title=f"{selected_player} - Fielding Contributions"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )