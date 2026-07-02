"""CricScope win-probability model evaluation (calibration + test metrics).

This script evaluates whichever model is configured for the Streamlit app:
- logistic: LogisticRegression + OneHotEncoder + passthrough numeric features
- random_forest: RandomForestClassifier
- xgboost: XGBClassifier

It computes on a held-out test split:
- accuracy
- log loss
- Brier score
- calibration / reliability curve (saved as an image)

Outputs:
- evaluation/results.json
- assets/calibration_curve_<model>.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'batting_team', 'bowling_team', 'city',
    'runs_left', 'balls_left', 'wickets',
    'target', 'crr', 'rrr'
]


def load_data(matches_path: str, deliveries_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def build_feature_matrix(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Replicates the feature engineering logic used in application.py (baseline model)."""
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = (
        df[df['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .reset_index()
        .rename(columns={'total_runs': 'target'})
    )
    total_df['target'] = total_df['target'] + 1

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2].copy()

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']

    balls_bowled = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    overs_bowled = (df['over'] - 1) + (df['ball'] / 6)
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Target: batting team wins
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[FEATURE_COLS + ['result', 'match_id']].dropna()
    return final_df


def build_preprocessor() -> ColumnTransformer:
    cat_features = ['batting_team', 'bowling_team', 'city']
    num_features = ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr']

    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ])


def build_model(model_name: str, seed: int = 42):
    if model_name == 'logistic':
        return LogisticRegression(max_iter=5000, random_state=seed)
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Compute reliability diagram stats."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_true_rates = np.zeros(n_bins, dtype=float)
    bin_pred_means = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        bin_counts[b] = int(mask.sum())
        if bin_counts[b] > 0:
            bin_true_rates[b] = y_true[mask].mean()
            bin_pred_means[b] = y_prob[mask].mean()
        else:
            bin_true_rates[b] = np.nan
            bin_pred_means[b] = np.nan

    return bins, bin_pred_means, bin_true_rates, bin_counts


@dataclass
class EvalResults:
    model: str
    n_bins: int
    split: str
    test_size: float
    seed: int
    n_test: int

    accuracy: float
    log_loss: float
    brier_score: float

    # Calibration
    calib_bin_pred_mean: list[float]
    calib_bin_true_rate: list[float]
    calib_bin_count: list[int]


def reliability_plot(bin_pred_mean, bin_true_rate, title: str, out_path: str):
    plt.figure(figsize=(7, 6))
    x = np.array(bin_pred_mean, dtype=float)
    y = np.array(bin_true_rate, dtype=float)

    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    plt.plot(x, y, marker="o", linewidth=2, label="Model")
    plt.xlabel("Predicted win probability")
    plt.ylabel("Observed win rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['logistic', 'random_forest', 'xgboost'], default='logistic')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=['match_level', 'row_level'], default='match_level')
    parser.add_argument('--n-bins', type=int, default=10)
    parser.add_argument('--matches-path', type=str, default='matches.csv')
    parser.add_argument('--deliveries-path', type=str, default='deliveries.csv')
    args = parser.parse_args()

    matches, deliveries = load_data(args.matches_path, args.deliveries_path)
    df = build_feature_matrix(deliveries, matches)

    X = df[FEATURE_COLS]
    y = df['result'].astype(int)

    if args.split == 'match_level':
        unique_matches = df['match_id'].unique()
        rng = np.random.default_rng(args.seed)
        n_test_matches = int(len(unique_matches) * args.test_size)
        test_matches = rng.choice(unique_matches, size=n_test_matches, replace=False)
        test_mask = df['match_id'].isin(test_matches).values
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(df))
        n_test = int(len(df) * args.test_size)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True

    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    preprocessor = build_preprocessor()
    model = build_model(args.model, seed=args.seed)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    ll = float(log_loss(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    _, bin_pred_mean, bin_true_rate, bin_counts = calibration_curve(y_test.values, y_prob, n_bins=args.n_bins)

    title = "Calibration Curve - Logistic Regression"
    out_png = f"assets/calibration_curve_{args.model}.png"
    reliability_plot(bin_pred_mean, bin_true_rate, title=title, out_path=out_png)

    results = EvalResults(
        model=args.model,
        n_bins=args.n_bins,
        split=args.split,
        test_size=args.test_size,
        seed=args.seed,
        n_test=int(len(X_test)),
        accuracy=acc,
        log_loss=ll,
        brier_score=brier,
        calib_bin_pred_mean=[(None if np.isnan(v) else float(v)) for v in bin_pred_mean],
        calib_bin_true_rate=[(None if np.isnan(v) else float(v)) for v in bin_true_rate],
        calib_bin_count=[int(c) for c in bin_counts],
    )

    os.makedirs('evaluation', exist_ok=True)
    out_json = 'evaluation/results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(asdict(results), f, indent=2)

    print(json.dumps(asdict(results), indent=2))


if __name__ == '__main__':
    main()

