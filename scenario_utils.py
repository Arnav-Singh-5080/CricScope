from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "city",
    "runs_left",
    "balls_left",
    "wickets",
    "total_runs_x",
    "crr",
    "rrr",
]


@st.cache_data
def load_matches(path: str = "matches.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_deliveries(path: str = "deliveries.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def get_team_players(deliveries: pd.DataFrame, batting_team: str) -> list:
    team_players = deliveries[deliveries["batting_team"] == batting_team]["batsman"].dropna().unique().tolist()
    return sorted(team_players)


def get_current_batsmen(deliveries: pd.DataFrame, batting_team: str) -> list:
    innings = deliveries[deliveries["inning"] == 2].copy()
    innings = innings[innings["batting_team"] == batting_team]

    dismissed = set(innings[innings["player_dismissed"].notna()]["player_dismissed"].astype(str).tolist())
    batting_card = (
        innings.groupby("batsman", as_index=False)
        .agg(balls_faced=("ball", "count"), runs_scored=("batsman_runs", "sum"))
        .sort_values(["balls_faced", "runs_scored"], ascending=False)
    )

    current = batting_card[~batting_card["batsman"].isin(dismissed)]["batsman"].head(2).tolist()
    if len(current) < 2:
        current = batting_card["batsman"].head(2).tolist()
    return current


def get_player_profile(deliveries: pd.DataFrame, batting_team: str, player: str) -> dict:
    player_df = deliveries[(deliveries["batting_team"] == batting_team) & (deliveries["batsman"] == player)].copy()
    if player_df.empty:
        return {
            "player": player,
            "balls_faced": 0,
            "runs_scored": 0,
            "strike_rate": 0.0,
            "average": 0.0,
            "boundaries": 0,
            "boundary_rate": 0.0,
            "dismissals": 0,
            "powerplay_sr": 0.0,
            "middle_sr": 0.0,
            "death_sr": 0.0,
            "role": "support batter",
        }

    balls_faced = int(player_df.shape[0])
    runs_scored = int(player_df["batsman_runs"].sum())
    dismissals = int((player_df["player_dismissed"] == player).sum())
    boundaries = int((player_df["batsman_runs"] >= 4).sum())

    strike_rate = round((runs_scored / balls_faced) * 100, 1) if balls_faced > 0 else 0.0
    average = round(runs_scored / dismissals, 1) if dismissals > 0 else float(runs_scored)
    boundary_rate = round(boundaries / balls_faced, 3) if balls_faced > 0 else 0.0

    def _phase_sr(df: pd.DataFrame) -> float:
        phase_balls = int(df.shape[0])
        if phase_balls == 0:
            return 0.0
        phase_runs = int(df["batsman_runs"].sum())
        return round((phase_runs / phase_balls) * 100, 1)

    powerplay_sr = _phase_sr(player_df[player_df["over"] <= 6])
    middle_sr = _phase_sr(player_df[(player_df["over"] >= 7) & (player_df["over"] <= 15)])
    death_sr = _phase_sr(player_df[player_df["over"] >= 16])

    if balls_faced < 20:
        role = "new batter"
    elif death_sr >= 140 or strike_rate >= 135:
        role = "finisher"
    elif strike_rate <= 110 and balls_faced >= 40:
        role = "anchor"
    elif strike_rate >= 125 or boundary_rate >= 0.20:
        role = "aggressor"
    else:
        role = "support batter"

    return {
        "player": player,
        "balls_faced": balls_faced,
        "runs_scored": runs_scored,
        "strike_rate": strike_rate,
        "average": average,
        "boundaries": boundaries,
        "boundary_rate": boundary_rate,
        "dismissals": dismissals,
        "powerplay_sr": powerplay_sr,
        "middle_sr": middle_sr,
        "death_sr": death_sr,
        "role": role,
    }


@st.cache_data
def get_player_stats(player_name: str, batting_team: str, deliveries: pd.DataFrame) -> dict | None:
    player_df = deliveries[
        (deliveries["batsman"] == player_name) &
        (deliveries["batting_team"] == batting_team)
    ].copy()

    if player_df.empty:
        return None

    total_runs = int(player_df["batsman_runs"].sum())
    total_balls = int(len(player_df))
    innings = int(player_df["match_id"].nunique())
    average = round(total_runs / max(innings, 1), 1)
    strike_rate = round((total_runs / max(total_balls, 1)) * 100, 1)

    match_ids_big = player_df.groupby("match_id")["batsman_runs"].sum()
    big_innings = match_ids_big[match_ids_big >= 20].index.tolist()

    boundaries = int((player_df["batsman_runs"] >= 4).sum())
    boundary_rate = round(boundaries / max(total_balls, 1), 3)

    powerplay_df = player_df[player_df["over"] <= 6]
    middle_df = player_df[(player_df["over"] >= 7) & (player_df["over"] <= 15)]
    death_df = player_df[player_df["over"] >= 16]

    def _phase_sr(df: pd.DataFrame) -> float:
        balls = int(len(df))
        if balls == 0:
            return 0.0
        runs = int(df["batsman_runs"].sum())
        return round((runs / balls) * 100, 1)

    powerplay_sr = _phase_sr(powerplay_df)
    middle_sr = _phase_sr(middle_df)
    death_sr = _phase_sr(death_df)

    if total_balls < 20:
        role = "new batter"
    elif death_sr >= 140 or strike_rate >= 135:
        role = "finisher"
    elif strike_rate <= 110 and total_balls >= 40:
        role = "anchor"
    elif strike_rate >= 125 or boundary_rate >= 0.2:
        role = "aggressor"
    else:
        role = "support batter"

    return {
        "name": player_name,
        "average": average,
        "strike_rate": strike_rate,
        "total_runs": total_runs,
        "innings": innings,
        "big_innings": len(big_innings),
        "balls_faced": total_balls,
        "boundary_rate": boundary_rate,
        "powerplay_sr": powerplay_sr,
        "middle_sr": middle_sr,
        "death_sr": death_sr,
        "role": role,
    }


def get_player_wicket_penalty(profile: dict) -> float:
    balls = float(profile.get("balls_faced", 0))
    strike_rate = float(profile.get("strike_rate", 0.0))
    boundary_rate = float(profile.get("boundary_rate", 0.0))
    death_sr = float(profile.get("death_sr", 0.0))
    role = str(profile.get("role", "support batter"))

    penalty = 0.06
    penalty += min(balls / 500.0, 0.05)
    penalty += max(strike_rate - 120.0, 0.0) / 700.0
    penalty += max(death_sr - 120.0, 0.0) / 900.0
    penalty += boundary_rate * 0.12

    role_adjustment = {
        "finisher": 0.07,
        "aggressor": 0.04,
        "anchor": -0.02,
        "support batter": 0.0,
        "new batter": -0.01,
    }
    penalty += role_adjustment.get(role, 0.0)
    return float(np.clip(penalty, 0.04, 0.25))


def calculate_player_impact(player_name: str, batting_team: str, balls_left: int, base_prob: float, deliveries: pd.DataFrame) -> tuple[float, float]:
    stats = get_player_stats(player_name, batting_team, deliveries)

    if stats is None:
        return round(base_prob, 3), 0.0

    expected_contribution = round(
        (stats["strike_rate"] / 100.0) *
        min(balls_left, (stats["average"] * 6.0 / max(stats["strike_rate"], 1.0) * 100.0)),
        1,
    )

    impact_weight = (
        stats["average"] * 0.6 +
        stats["strike_rate"] * 0.4
    ) / 100.0

    phase_boost = 0.0
    if stats["role"] == "finisher":
        phase_boost = 0.05
    elif stats["role"] == "aggressor":
        phase_boost = 0.03
    elif stats["role"] == "anchor":
        phase_boost = -0.01

    adjusted_drop = min((impact_weight * 0.35) + phase_boost, 0.30)
    adjusted_drop += min(expected_contribution / 1000.0, 0.02)

    scenario_prob = max(base_prob - adjusted_drop, 0.02)
    return round(scenario_prob, 3), round(adjusted_drop * 100, 1)


def get_player_explanation(profile: dict) -> str:
    player = profile.get("player", "This batter")
    role = profile.get("role", "support batter")
    strike_rate = profile.get("strike_rate", 0.0)
    boundary_rate = profile.get("boundary_rate", 0.0)
    balls_faced = profile.get("balls_faced", 0)
    death_sr = profile.get("death_sr", 0.0)

    if role == "finisher":
        return (
            f"{player} is a finisher. He usually scores quickly at the end, so losing him hurts the chase more than usual. "
            f"He has a strike rate of {strike_rate:.0f} and a strong death-overs record of {death_sr:.0f}."
        )
    if role == "aggressor":
        return (
            f"{player} likes to attack the bowlers. He scores at a strike rate of {strike_rate:.0f}, so taking him out removes pressure from the attack."
        )
    if role == "anchor":
        return (
            f"{player} is an anchor. He keeps the innings steady, so the drop is smaller than for a quick scorer."
        )
    if role == "new batter":
        return (
            f"{player} has not faced many balls for this team yet, so the effect is usually modest."
        )

    return (
        f"{player} gives steady support. He has faced {balls_faced} balls, a strike rate of {strike_rate:.0f}, and a boundary rate of {boundary_rate:.2f}, so losing him still matters."
    )


def _normalize_numeric(value: float) -> float:
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def build_feature_vector(
    batting_team: str,
    bowling_team: str,
    city: str,
    current_score: float,
    target: float,
    overs_done: float,
    wickets_fallen: float,
) -> pd.DataFrame:
    current_score = max(0.0, _normalize_numeric(current_score))
    target = max(0.0, _normalize_numeric(target))
    overs_done = max(0.0, _normalize_numeric(overs_done))
    wickets_fallen = min(max(0.0, _normalize_numeric(wickets_fallen)), 9.0)

    balls_left = max(0.0, 120.0 - (overs_done * 6.0))
    runs_left = max(0.0, target - current_score)
    crr = current_score / overs_done if overs_done > 0 else 0.0
    rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets": wickets_fallen,
                "total_runs_x": target,
                "crr": crr,
                "rrr": rrr,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


@st.cache_resource
def load_pipeline(path: str = "pipe.pkl"):
    import joblib
    import sklearn.compose._column_transformer as column_transformer_module

    if not hasattr(column_transformer_module, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer_module._RemainderColsList = _RemainderColsList

    return joblib.load(path)


def _prepare_model_matrix(pipe: Any, feature_df: pd.DataFrame):
    preprocessor = pipe.steps[0][1]
    model = pipe.steps[-1][1]

    feature_order = list(getattr(pipe, "feature_names_in_", FEATURE_COLUMNS))
    ordered_df = feature_df[feature_order]

    transformed = preprocessor.transform(ordered_df)
    transformed_matrix = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    expected_features = getattr(model, "coef_", np.empty((1, transformed_matrix.shape[1]))).shape[1]
    if transformed_matrix.shape[1] < expected_features:
        numeric_columns = ["runs_left", "balls_left", "wickets", "total_runs_x", "crr", "rrr"]
        numeric_values = ordered_df[numeric_columns].to_numpy(dtype=float)
        transformed_matrix = np.hstack([transformed_matrix, numeric_values])

    return transformed_matrix, preprocessor, model, ordered_df


def get_win_probability(pipe: Any, feature_df: pd.DataFrame) -> float:
    model_matrix, _, model, _ = _prepare_model_matrix(pipe, feature_df)
    return float(model.predict_proba(model_matrix)[0][1])


def compute_delta(base_prob: float, scenario_prob: float) -> dict:
    delta = scenario_prob - base_prob
    if delta > 0:
        direction = "improved"
        label = "↑ Improved"
    elif delta < 0:
        direction = "worsened"
        label = "↓ Worsened"
    else:
        direction = "neutral"
        label = "— No change"

    confidence_source = max(base_prob, scenario_prob)
    if confidence_source >= 0.7:
        confidence_tier = "High"
    elif confidence_source >= 0.5:
        confidence_tier = "Moderate"
    else:
        confidence_tier = "Close"

    return {
        "delta": delta,
        "direction": direction,
        "confidence_tier": confidence_tier,
        "label": label,
    }


def run_shap(pipe: Any, feature_df: pd.DataFrame) -> tuple:
    model_matrix, preprocessor, model, ordered_df = _prepare_model_matrix(pipe, feature_df)

    feature_names = list(preprocessor.get_feature_names_out())
    numeric_names = ["runs_left", "balls_left", "wickets", "total_runs_x", "crr", "rrr"]
    if len(feature_names) < model_matrix.shape[1]:
        feature_names.extend(numeric_names[: model_matrix.shape[1] - len(feature_names)])

    try:
        import shap

        explainer = shap.LinearExplainer(model, model_matrix)
        shap_values = explainer.shap_values(model_matrix)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 2:
            shap_array = shap_array[0]

        return shap_array, feature_names
    except Exception:
        coefficients = np.asarray(getattr(model, "coef_", np.zeros((1, model_matrix.shape[1]))))[0]
        baseline = np.asarray(getattr(model, "intercept_", np.zeros(1)))[0]
        contributions = model_matrix[0] * coefficients
        shap_array = np.asarray(contributions)
        if shap_array.ndim == 0:
            shap_array = np.array([float(shap_array)])

        # Keep the fallback deterministic and centered around the model logit.
        if shap_array.shape[0] < len(feature_names):
            padding = np.zeros(len(feature_names) - shap_array.shape[0])
            shap_array = np.concatenate([shap_array, padding])

    return shap_array, feature_names


def predict_score_range(current_score: float, crr: float, balls_left: float) -> dict:
    balls_left = max(0.0, _normalize_numeric(balls_left))
    crr = max(0.0, _normalize_numeric(crr))
    current_score = max(0.0, _normalize_numeric(current_score))

    predicted_add = crr * (balls_left / 6.0)
    return {
        "low": round(current_score + predicted_add * 0.8, 1),
        "mid": round(current_score + predicted_add, 1),
        "high": round(current_score + predicted_add * 1.2, 1),
    }


def build_momentum_series(pipe: Any, base_inputs: dict, scenario_inputs: dict, split_over: float) -> tuple:
    def _series(inputs: dict) -> list:
        overs_done = max(0.1, _normalize_numeric(inputs.get("overs_done", split_over)))
        current_score = max(0.0, _normalize_numeric(inputs.get("current_score", 0.0)))
        target = max(0.0, _normalize_numeric(inputs.get("target", 0.0)))
        wickets_fallen = min(max(0.0, _normalize_numeric(inputs.get("wickets_fallen", 0.0))), 9.0)
        batting_team = str(inputs.get("batting_team", ""))
        bowling_team = str(inputs.get("bowling_team", ""))
        city = str(inputs.get("city", ""))

        balls_left = max(0.0, 120.0 - (overs_done * 6.0))
        crr = current_score / overs_done if overs_done > 0 else 0.0
        projection = predict_score_range(current_score, crr, balls_left)["mid"]

        series = []
        for over in range(1, 21):
            if over <= split_over:
                progress = over / max(split_over, 0.1)
                score = current_score * progress
                wickets = min(9.0, wickets_fallen * progress)
            else:
                progress = (over - split_over) / max(20.0 - split_over, 1.0)
                score = current_score + (projection - current_score) * progress
                wickets = min(9.0, wickets_fallen + (2.0 * progress))

            feature_df = build_feature_vector(
                batting_team=batting_team,
                bowling_team=bowling_team,
                city=city,
                current_score=score,
                target=target,
                overs_done=float(over),
                wickets_fallen=wickets,
            )
            series.append(get_win_probability(pipe, feature_df) * 100.0)

        return series

    return _series(base_inputs), _series(scenario_inputs)


def get_batting_pair(deliveries: pd.DataFrame, match_state: dict) -> list:
    subset = deliveries.copy()
    batting_team = match_state.get("batting_team")
    bowling_team = match_state.get("bowling_team")

    if "inning" in subset.columns:
        subset = subset[subset["inning"] == 2]
    if batting_team and "batting_team" in subset.columns:
        subset = subset[subset["batting_team"] == batting_team]
    if bowling_team and "bowling_team" in subset.columns:
        subset = subset[subset["bowling_team"] == bowling_team]

    batters = subset.get("batsman", pd.Series(dtype=str)).dropna().value_counts().index.tolist()
    if not batters:
        batters = deliveries.get("batsman", pd.Series(dtype=str)).dropna().value_counts().index.tolist()
    return batters[:2] if len(batters) >= 2 else batters


def get_available_batsmen(deliveries: pd.DataFrame, match_state: dict) -> list:
    subset = deliveries.copy()
    batting_team = match_state.get("batting_team")
    dismissed_batsman = match_state.get("dismissed_batsman")

    if batting_team and "batting_team" in subset.columns:
        subset = subset[subset["batting_team"] == batting_team]
    if "inning" in subset.columns:
        subset = subset[subset["inning"] == 2]

    batters = subset.get("batsman", pd.Series(dtype=str)).dropna().value_counts().index.tolist()
    if dismissed_batsman in batters:
        batters.remove(dismissed_batsman)
    return batters


def get_available_bowlers(deliveries: pd.DataFrame, match_state: dict) -> list:
    subset = deliveries.copy()
    bowling_team = match_state.get("bowling_team")

    if bowling_team and "bowling_team" in subset.columns:
        subset = subset[subset["bowling_team"] == bowling_team]
    if "inning" in subset.columns:
        subset = subset[subset["inning"] == 2]

    bowlers = subset.get("bowler", pd.Series(dtype=str)).dropna().value_counts().index.tolist()
    if not bowlers:
        bowlers = deliveries.get("bowler", pd.Series(dtype=str)).dropna().value_counts().index.tolist()
    return bowlers
