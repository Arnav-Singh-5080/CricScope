import pandas as pd
import numpy as np

# Dismissals that count against the BOWLER for bowling average / wickets.
# run_out, retired_hurt, obstructing_the_field do NOT count as bowler wickets.
BOWLER_WICKET_KINDS = {
    "bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"
}


def _season_year(season_str: str) -> int:
    """'IPL-2017' -> 2017. Handles plain '2017' too, just in case."""
    return int(str(season_str).split("-")[-1])


def get_all_players(deliveries: pd.DataFrame) -> list[str]:
    """Sorted list of every player who has either batted or bowled.
    Use this to populate the searchable dropdown."""
    batters = set(deliveries["batsman"].dropna().unique())
    bowlers = set(deliveries["bowler"].dropna().unique())
    return sorted(batters | bowlers)


def _with_season(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join deliveries to matches on match_id to attach season year.
    Do this ONCE outside the per-player functions if you're calling them
    in a loop — see merge_once() below."""
    merged = deliveries.merge(
        matches[["id", "Season"]], left_on="match_id", right_on="id", how="left"
    )
    merged["season_year"] = merged["Season"].apply(_season_year)
    return merged


def merge_once(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Call this ONCE at app startup (e.g. cached with @st.cache_data) and
    pass the result into every other function below as `df`.
    This avoids re-merging 179k rows on every dropdown selection."""
    return _with_season(deliveries, matches)


def get_career_batting_stats(df: pd.DataFrame, player: str) -> dict:
    """df = output of merge_once(). Returns career-level batting numbers."""
    bat = df[df["batsman"] == player]

    if bat.empty:
        return {
            "matches": 0, "innings": 0, "runs": 0, "balls_faced": 0,
            "average": None, "strike_rate": None, "highest_score": 0,
        }

    matches_played = bat["match_id"].nunique()

    # innings batted = distinct (match_id, inning) pairs where they faced a ball
    innings = bat.groupby(["match_id", "inning"]).size().shape[0]

    runs = int(bat["batsman_runs"].sum())

    # balls faced excludes wides (wide deliveries are not "faced" by the batsman)
    balls_faced = int((bat["wide_runs"] == 0).sum())

    # dismissals: count how many times THIS player is in player_dismissed
    dismissals = int((df["player_dismissed"] == player).sum())
    not_outs = innings - dismissals

    average = round(runs / dismissals, 2) if dismissals > 0 else None
    strike_rate = round((runs / balls_faced) * 100, 2) if balls_faced > 0 else None

    # highest score per innings, then take the max
    runs_per_innings = bat.groupby(["match_id", "inning"])["batsman_runs"].sum()
    highest_score = int(runs_per_innings.max()) if not runs_per_innings.empty else 0

    return {
        "matches": matches_played,
        "innings": innings,
        "runs": runs,
        "balls_faced": balls_faced,
        "not_outs": not_outs,
        "dismissals": dismissals,
        "average": average,
        "strike_rate": strike_rate,
        "highest_score": highest_score,
    }


def get_career_bowling_stats(df: pd.DataFrame, player: str) -> dict:
    """df = output of merge_once(). Returns career-level bowling numbers."""
    bowl = df[df["bowler"] == player]

    if bowl.empty:
        return {
            "innings_bowled": 0, "balls_bowled": 0, "runs_conceded": 0,
            "wickets": 0, "average": None, "strike_rate": None, "economy": None,
        }

    innings_bowled = bowl.groupby(["match_id", "inning"]).size().shape[0]

    # legal deliveries only: wides and no-balls are not counted toward the
    # bowler's ball count (standard cricket scoring rule)
    legal_balls = bowl[(bowl["wide_runs"] == 0) & (bowl["noball_runs"] == 0)]
    balls_bowled = int(legal_balls.shape[0])

    # runs conceded by bowler = total_runs minus byes/leg-byes (those aren't
    # the bowler's fault) but wides/no-balls DO count against the bowler
    runs_conceded = int(
        (bowl["total_runs"] - bowl["bye_runs"] - bowl["legbye_runs"]).sum()
    )

    wickets = int(
        bowl[bowl["dismissal_kind"].isin(BOWLER_WICKET_KINDS)].shape[0]
    )

    average = round(runs_conceded / wickets, 2) if wickets > 0 else None
    strike_rate = round(balls_bowled / wickets, 2) if wickets > 0 else None
    economy = round(runs_conceded / (balls_bowled / 6), 2) if balls_bowled > 0 else None

    return {
        "innings_bowled": innings_bowled,
        "balls_bowled": balls_bowled,
        "runs_conceded": runs_conceded,
        "wickets": wickets,
        "average": average,
        "strike_rate": strike_rate,
        "economy": economy,
    }


def get_total_catches(df: pd.DataFrame, player: str) -> int:
    """Catches taken as a fielder. Only counts dismissal_kind == 'caught'
    (NOT 'caught and bowled', since that already shows in bowler wickets
    and crediting it again here would double count for an all-rounder)."""
    catches = df[
        (df["fielder"] == player) & (df["dismissal_kind"] == "caught")
    ]
    return int(catches.shape[0])


def get_career_summary(df: pd.DataFrame, player: str) -> dict:
    """The single function the UI page should call for the Career Summary
    card. Combines batting + bowling + fielding into one flat dict matching
    the fields requested in Issue #561."""
    bat = get_career_batting_stats(df, player)
    bowl = get_career_bowling_stats(df, player)
    catches = get_total_catches(df, player)

    # matches played = union of matches they batted OR bowled in
    bat_matches = set(df[df["batsman"] == player]["match_id"].unique())
    bowl_matches = set(df[df["bowler"] == player]["match_id"].unique())
    total_matches = len(bat_matches | bowl_matches)

    return {
        "player": player,
        "matches_played": total_matches,
        "innings": bat["innings"],
        "total_runs": bat["runs"],
        "batting_average": bat["average"],
        "strike_rate": bat["strike_rate"],
        "highest_score": bat["highest_score"],
        "total_wickets": bowl["wickets"],
        "bowling_average": bowl["average"],
        "bowling_strike_rate": bowl["strike_rate"],
        "economy_rate": bowl["economy"],
        "total_catches": catches,
    }


def get_team_history(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Returns a Season -> Team table. A player's 'team' for a season is
    the batting_team they appear under most often that season (handles
    the edge case of a player appearing for two teams in one season due
    to a mid-season trade, which does happen in IPL)."""
    bat_rows = df[df["batsman"] == player][["season_year", "batting_team"]]
    bowl_rows = df[df["bowler"] == player][["season_year", "bowling_team"]].rename(
        columns={"bowling_team": "batting_team"}
    )
    combined = pd.concat([bat_rows, bowl_rows], ignore_index=True)

    if combined.empty:
        return pd.DataFrame(columns=["Season", "Team"])

    team_per_season = (
        combined.groupby("season_year")["batting_team"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"season_year": "Season", "batting_team": "Team"})
        .sort_values("Season")
        .reset_index(drop=True)
    )
    return team_per_season


def get_season_wise_performance(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """One row per season the player was active, with batting + bowling
    numbers for that season only. Feeds both the table and the 3 charts
    (Runs vs Season, SR vs Season, Wickets vs Season)."""
    seasons = sorted(
        set(df[df["batsman"] == player]["season_year"].unique())
        | set(df[df["bowler"] == player]["season_year"].unique())
    )

    rows = []
    for yr in seasons:
        season_df = df[df["season_year"] == yr]
        bat = get_career_batting_stats(season_df, player)
        bowl = get_career_bowling_stats(season_df, player)
        rows.append({
            "Season": yr,
            "Runs": bat["runs"],
            "Batting_Average": bat["average"],
            "Strike_Rate": bat["strike_rate"],
            "Wickets": bowl["wickets"],
            "Economy_Rate": bowl["economy"],
        })

    return pd.DataFrame(rows)