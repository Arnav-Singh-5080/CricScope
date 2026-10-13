from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CAT_FEATURES = ["batting_team", "bowling_team", "city"]
NUM_FEATURES = ["runs_left", "balls_left", "wickets", "target", "crr", "rrr"]
FEATURE_COLUMNS = CAT_FEATURES + NUM_FEATURES


def calculate_brier_score(y_true, y_prob) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        return float("nan")

    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def classify_phase_from_balls_left(balls_left: float) -> str:
    overs_completed = (120 - balls_left) / 6

    if overs_completed <= 6:
        return "Powerplay"

    if overs_completed <= 15:
        return "Middle Overs"

    return "Death Overs"


def is_pressure_situation(row: pd.Series) -> bool:
    return (
        row["rrr"] > 10
        and row["balls_left"] < 30
        and row["wickets"] <= 4
        and row["runs_left"] > 40
    )


def prepare_match_state_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    matches_df = matches.copy()
    deliveries_df = deliveries.copy()

    required_delivery_columns = {
        "match_id",
        "inning",
        "batting_team",
        "bowling_team",
        "over",
        "ball",
        "total_runs",
        "player_dismissed",
    }

    missing_columns = required_delivery_columns - set(deliveries_df.columns)
    if missing_columns:
        raise ValueError(f"Missing delivery columns: {sorted(missing_columns)}")

    if "id" not in matches_df.columns:
        raise ValueError("matches.csv must contain an 'id' column")

    if "winner" not in matches_df.columns:
        raise ValueError("matches.csv must contain a 'winner' column")

    if "city" not in matches_df.columns:
        matches_df["city"] = matches_df.get("venue", "Unknown")

    if "season" not in matches_df.columns:
        matches_df["season"] = "Unknown"

    match_columns = ["id", "winner", "city", "season"]

    merged = deliveries_df.merge(
        matches_df[match_columns],
        left_on="match_id",
        right_on="id",
        how="left",
    )

    first_innings_totals = (
        merged[merged["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target"})
    )

    first_innings_totals["target"] = first_innings_totals["target"] + 1

    chase_df = merged.merge(first_innings_totals, on="match_id", how="inner")
    chase_df = chase_df[chase_df["inning"] == 2].copy()

    chase_df["current_score"] = chase_df.groupby("match_id")["total_runs"].cumsum()
    chase_df["runs_left"] = chase_df["target"] - chase_df["current_score"]

    balls_bowled = ((chase_df["over"] - 1) * 6) + chase_df["ball"]
    chase_df["balls_left"] = (120 - balls_bowled).clip(lower=0)

    chase_df["player_dismissed"] = chase_df["player_dismissed"].notna().astype(int)
    wickets_lost = chase_df.groupby("match_id")["player_dismissed"].cumsum()
    chase_df["wickets"] = 10 - wickets_lost

    overs_bowled = (chase_df["over"] - 1) + (chase_df["ball"] / 6)
    chase_df["crr"] = np.where(
        overs_bowled > 0,
        chase_df["current_score"] / overs_bowled,
        0.0,
    )

    chase_df["rrr"] = np.where(
        chase_df["balls_left"] > 0,
        (chase_df["runs_left"] * 6) / chase_df["balls_left"],
        0.0,
    )

    chase_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    chase_df["result"] = np.where(chase_df["batting_team"] == chase_df["winner"], 1, 0)
    chase_df["phase"] = chase_df["balls_left"].apply(classify_phase_from_balls_left)
    chase_df["pressure_situation"] = chase_df.apply(is_pressure_situation, axis=1)

    final_columns = [
        "match_id",
        "season",
        "batting_team",
        "bowling_team",
        "city",
        "runs_left",
        "balls_left",
        "wickets",
        "target",
        "crr",
        "rrr",
        "phase",
        "pressure_situation",
        "result",
    ]

    final_df = chase_df[final_columns].dropna().copy()
    final_df = final_df[final_df["balls_left"] >= 0]
    final_df = final_df[final_df["wickets"] >= 0]

    return final_df


def build_prediction_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CAT_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUM_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def run_historical_backtest(match_states: pd.DataFrame) -> pd.DataFrame:
    if match_states.empty:
        return pd.DataFrame()

    df = match_states.copy()
    df = df.sort_values(["season", "match_id"]).reset_index(drop=True)

    seasons = list(pd.Series(df["season"].unique()).dropna())

    backtest_frames = []

    if len(seasons) < 2:
        split_index = int(len(df) * 0.75)
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        if train_df.empty or test_df.empty:
            return pd.DataFrame()

        if train_df["result"].nunique() < 2:
            return pd.DataFrame()

        pipeline = build_prediction_pipeline()
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["result"])

        evaluated = test_df.copy()
        evaluated["win_probability"] = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        evaluated["predicted_result"] = (evaluated["win_probability"] >= 0.5).astype(int)
        evaluated["correct_prediction"] = (
            evaluated["predicted_result"] == evaluated["result"]
        ).astype(int)

        return evaluated

    for season in seasons[1:]:
        train_df = df[df["season"] < season]
        test_df = df[df["season"] == season]

        if train_df.empty or test_df.empty:
            continue

        if train_df["result"].nunique() < 2:
            continue

        pipeline = build_prediction_pipeline()
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["result"])

        evaluated = test_df.copy()
        evaluated["win_probability"] = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        evaluated["predicted_result"] = (evaluated["win_probability"] >= 0.5).astype(int)
        evaluated["correct_prediction"] = (
            evaluated["predicted_result"] == evaluated["result"]
        ).astype(int)

        backtest_frames.append(evaluated)

    if not backtest_frames:
        return pd.DataFrame()

    return pd.concat(backtest_frames, ignore_index=True)


def calculate_model_summary(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "samples": 0,
            "accuracy": float("nan"),
            "brier_score": float("nan"),
            "average_predicted_probability": float("nan"),
            "actual_win_rate": float("nan"),
            "average_confidence": float("nan"),
        }

    confidence = np.maximum(
        predictions["win_probability"],
        1 - predictions["win_probability"],
    )

    return {
        "samples": int(len(predictions)),
        "accuracy": float(predictions["correct_prediction"].mean()),
        "brier_score": calculate_brier_score(
            predictions["result"],
            predictions["win_probability"],
        ),
        "average_predicted_probability": float(predictions["win_probability"].mean()),
        "actual_win_rate": float(predictions["result"].mean()),
        "average_confidence": float(confidence.mean()),
    }


def create_probability_buckets(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "prediction_bucket",
                "sample_count",
                "average_predicted_probability",
                "actual_win_rate",
                "calibration_gap",
            ]
        )

    df = predictions.copy()

    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}%" for i in range(n_bins)]

    df["prediction_bucket"] = pd.cut(
        df["win_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grouped = (
        df.groupby("prediction_bucket", observed=False)
        .agg(
            sample_count=("result", "size"),
            average_predicted_probability=("win_probability", "mean"),
            actual_win_rate=("result", "mean"),
        )
        .reset_index()
    )

    grouped["calibration_gap"] = (
        grouped["actual_win_rate"] - grouped["average_predicted_probability"]
    )

    return grouped


def calculate_season_wise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return _group_metrics(predictions, "season")


def calculate_phase_wise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return _group_metrics(predictions, "phase")


def calculate_pressure_situation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    pressure_df = predictions[predictions["pressure_situation"]].copy()

    if pressure_df.empty:
        return pd.DataFrame(
            [
                {
                    "situation": "Pressure Chase",
                    "samples": 0,
                    "accuracy": float("nan"),
                    "brier_score": float("nan"),
                    "average_predicted_probability": float("nan"),
                    "actual_win_rate": float("nan"),
                    "calibration_gap": float("nan"),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "situation": "Pressure Chase",
                "samples": int(len(pressure_df)),
                "accuracy": float(pressure_df["correct_prediction"].mean()),
                "brier_score": calculate_brier_score(
                    pressure_df["result"],
                    pressure_df["win_probability"],
                ),
                "average_predicted_probability": float(
                    pressure_df["win_probability"].mean()
                ),
                "actual_win_rate": float(pressure_df["result"].mean()),
                "calibration_gap": float(
                    pressure_df["result"].mean()
                    - pressure_df["win_probability"].mean()
                ),
            }
        ]
    )


def _group_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if predictions.empty or group_column not in predictions.columns:
        return pd.DataFrame()

    rows = []

    for group_value, group_df in predictions.groupby(group_column):
        rows.append(
            {
                group_column: group_value,
                "samples": int(len(group_df)),
                "accuracy": float(group_df["correct_prediction"].mean()),
                "brier_score": calculate_brier_score(
                    group_df["result"],
                    group_df["win_probability"],
                ),
                "average_predicted_probability": float(
                    group_df["win_probability"].mean()
                ),
                "actual_win_rate": float(group_df["result"].mean()),
                "calibration_gap": float(
                    group_df["result"].mean()
                    - group_df["win_probability"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)
