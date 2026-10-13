import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Player Comparison",
    page_icon="🆚",
    layout="wide"
)

@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

players = sorted(
    deliveries["batsman"].dropna().unique()
)

st.title("🆚 IPL Player Comparison")
col1, col2 = st.columns(2)

with col1:
    player1 = st.selectbox(
        "Player 1",
        players,
        index=0
    )

with col2:
    player2 = st.selectbox(
        "Player 2",
        players,
        index=1
    )
    

def get_stats(player):

    pdata = deliveries[
        deliveries["batsman"] == player
    ]

    runs = pdata["batsman_runs"].sum()

    balls = len(pdata)

    sr = (
        round((runs / balls) * 100, 2)
        if balls > 0 else 0
    )

    matches_played = pdata["match_id"].nunique()

    match_scores = (
        pdata.groupby("match_id")["batsman_runs"]
        .sum()
    )

    highest = (
        int(match_scores.max())
        if not match_scores.empty else 0
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

    return {
        "Matches": matches_played,
        "Runs": runs,
        "SR": sr,
        "HS": highest,
        "50s": fifties,
        "100s": hundreds
    }
stats1 = get_stats(player1)
stats2 = get_stats(player2)

comparison = pd.DataFrame({
    player1: stats1,
    player2: stats2
})

st.dataframe(
    comparison,
    use_container_width=True
)    