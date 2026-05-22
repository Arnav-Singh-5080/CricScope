"""
Retrain the IPL win-probability XGBoost model (same pipeline as XGBOOST.ipynb).

Usage:
  python train_model.py              # fast retrain with known best hyperparameters
  python train_model.py --tune       # full GridSearchCV (~15+ minutes)
  python train_model.py --compare    # evaluate old model on new test split before saving
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "xgb_win_prob_model.json"
OLD_MODEL_PATH = ROOT / "xgb_win_prob_model.json.bak"
METRICS_PATH = ROOT / "model_metrics.json"

BEST_PARAMS = {
    "gamma": 0,
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 200,
    "reg_alpha": 1,
    "reg_lambda": 1,
}


def rename_team_cols(col_name: str) -> str:
    if col_name.startswith("batting_team_"):
        return col_name.replace("batting_team_", "bat_")
    if col_name.startswith("bowling_team_"):
        return col_name.replace("bowling_team_", "bowl_")
    return col_name


def build_feature_matrix(matches_path: Path, deliveries_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    total_score_df = deliveries.groupby(["match_id", "inning"])["total_runs"].sum().reset_index()
    total_score_df = total_score_df[total_score_df["inning"] == 1]
    total_score_df["target_score"] = total_score_df["total_runs"] + 1
    total_score_df = total_score_df[["match_id", "target_score"]]

    match_df = matches.merge(total_score_df, left_on="id", right_on="match_id")
    match_df = match_df[["match_id", "team1", "team2", "winner", "target_score"]]

    delivery_df = deliveries[deliveries["inning"] == 2].copy()
    delivery_df = delivery_df.merge(match_df, on="match_id")

    delivery_df["current_score"] = delivery_df.groupby("match_id")["total_runs"].cumsum()
    delivery_df["runs_left"] = (delivery_df["target_score"] - delivery_df["current_score"]).clip(lower=0)

    delivery_df["balls_bowled"] = (delivery_df["over"] - 1) * 6 + delivery_df["ball"]
    delivery_df["balls_left"] = (120 - delivery_df["balls_bowled"]).clip(lower=0)

    delivery_df["player_dismissed"] = delivery_df["player_dismissed"].fillna("0")
    delivery_df["player_dismissed"] = delivery_df["player_dismissed"].apply(
        lambda x: x if x == "0" else "1"
    ).astype(int)
    delivery_df["wickets_fallen"] = delivery_df.groupby("match_id")["player_dismissed"].cumsum()
    delivery_df["wickets_left"] = 10 - delivery_df["wickets_fallen"]

    delivery_df["result"] = (delivery_df["batting_team"] == delivery_df["winner"]).astype(int)
    delivery_df["crr"] = np.where(
        delivery_df["balls_bowled"] > 0,
        (delivery_df["current_score"] * 6) / delivery_df["balls_bowled"],
        0,
    )
    delivery_df["rrr"] = np.where(
        delivery_df["balls_left"] > 0,
        (delivery_df["runs_left"] * 6) / delivery_df["balls_left"],
        0,
    )

    features_to_keep = [
        "batting_team",
        "bowling_team",
        "target_score",
        "runs_left",
        "balls_left",
        "crr",
        "rrr",
        "wickets_left",
        "result",
    ]
    df_model = delivery_df[features_to_keep].copy()
    df_encoded = pd.get_dummies(df_model, columns=["batting_team", "bowling_team"], dtype=int)

    X = df_encoded.drop(columns=["result"])
    y = df_encoded["result"]
    X.columns = [rename_team_cols(c) for c in X.columns]
    return X, y


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return X.reindex(columns=feature_names, fill_value=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV instead of fixed best params")
    parser.add_argument("--compare", action="store_true", help="Score previous model before overwriting")
    args = parser.parse_args()

    print("Building features from matches.csv and deliveries.csv ...")
    X, y = build_feature_matrix(ROOT / "matches.csv", ROOT / "deliveries.csv")
    print(f"Samples: {len(X)}, features: {len(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    metrics: dict = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(X.columns)),
        "features": list(X.columns),
    }

    old_test_acc = None
    if args.compare and MODEL_PATH.exists():
        old_model = xgb.XGBClassifier()
        old_model.load_model(str(MODEL_PATH))
        old_features = list(old_model.feature_names_in_)
        X_old = align_features(X_test, old_features)
        old_test_acc = float(accuracy_score(y_test, old_model.predict(X_old)))
        metrics["old_model_test_accuracy"] = old_test_acc
        print(f"Old model test accuracy (aligned features): {old_test_acc:.4f}")

    if args.tune:
        print("Starting GridSearchCV (this may take 15+ minutes) ...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 5, 10],
            "gamma": [0, 0.1, 0.5],
        }
        grid = GridSearchCV(
            xgb.XGBClassifier(eval_metric="logloss", random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        start = time.time()
        grid.fit(X_train, y_train)
        print(f"Tuning done in {time.time() - start:.1f}s")
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        model = grid.best_estimator_
        metrics["best_params"] = grid.best_params_
        metrics["cv_accuracy"] = float(grid.best_score_)
    else:
        model = xgb.XGBClassifier(eval_metric="logloss", random_state=42, **BEST_PARAMS)
        model.fit(X_train, y_train)
        metrics["best_params"] = BEST_PARAMS

    test_acc = float(accuracy_score(y_test, model.predict(X_test)))
    metrics["new_model_test_accuracy"] = test_acc
    print(f"New model test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    if MODEL_PATH.exists():
        import shutil

        shutil.copy2(MODEL_PATH, OLD_MODEL_PATH)
        print(f"Backed up model -> {OLD_MODEL_PATH.name}")

    model.save_model(str(MODEL_PATH))
    X_test.to_csv(ROOT / "X_test.csv", index=False)
    X_train.to_csv(ROOT / "X_train.csv", index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH.name}, X_train.csv, X_test.csv, {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
