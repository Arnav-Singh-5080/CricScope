"""Improved ML pipeline for CricScope win prediction.

Improvements over baseline:
1. New features: innings phase, pressure index, run rate momentum,
   wicket resource value, venue chase rate, toss advantage
2. GroupKFold cross-validation at match level (prevents data leakage)
3. XGBoost hyperparameter tuning via RandomizedSearchCV
4. Ensemble: LR + RF + XGB with soft voting
5. Full evaluation: accuracy, precision, recall, F1, ROC-AUC, calibration
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ── Feature Engineering ──────────────────────────────────────────────────────

def add_innings_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Add powerplay / middle / death overs phase."""
    df = df.copy()
    df['innings_phase'] = pd.cut(
        df['over'],
        bins=[0, 6, 15, 20],
        labels=['powerplay', 'middle', 'death']
    ).astype(str)
    return df


def add_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    """Pressure = RRR / CRR ratio. >1.5 = high pressure on batting team."""
    df = df.copy()
    df['pressure_index'] = np.where(
        df['crr'] > 0,
        (df['rrr'] / df['crr']).clip(0, 5),
        df['rrr'].clip(0, 5)
    )
    return df


def add_wicket_resource(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resource value of remaining wickets (inspired by D/L method).
    More wickets in hand = exponentially more resources.
    """
    df = df.copy()
    df['wicket_resource'] = (df['wickets'] / 10) ** 1.5
    return df


def add_run_rate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """RRR - CRR: positive means batting team is behind, negative means ahead."""
    df = df.copy()
    df['rr_momentum'] = df['rrr'] - df['crr']
    return df


def add_balls_phase_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Balls remaining × wickets in hand — composite resource."""
    df = df.copy()
    df['balls_wickets'] = df['balls_left'] * df['wickets']
    return df


def add_venue_features(df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add venue-level chase win rate from historical matches data."""
    from model.feature_engineering import compute_venue_chase_win_rate, compute_toss_win_rate

    venue_chase = compute_venue_chase_win_rate(matches_df)
    venue_toss = compute_toss_win_rate(matches_df)

    df = df.copy()
    df['venue_chase_rate'] = df['city'].map(venue_chase).fillna(0.5)
    df['venue_toss_advantage'] = df['city'].map(venue_toss).fillna(0.5)
    return df


def build_feature_matrix(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build full feature matrix from raw ball-by-ball data."""
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # Target from 1st innings
    total_df = (
        df[df['inning'] == 1]
        .groupby('match_id')['total_runs'].sum()
        .reset_index()
        .rename(columns={'total_runs': 'target'})
    )
    total_df['target'] += 1

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2].copy()

    # Core features
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

    # Label
    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    # New features
    df = add_innings_phase(df)
    df = add_pressure_index(df)
    df = add_wicket_resource(df)
    df = add_run_rate_momentum(df)
    df = add_balls_phase_interaction(df)
    df = add_venue_features(df, matches)

    feature_cols = [
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets',
        'target', 'crr', 'rrr',
        # New features
        'innings_phase', 'pressure_index', 'wicket_resource',
        'rr_momentum', 'balls_wickets',
        'venue_chase_rate', 'venue_toss_advantage',
        'result', 'match_id'
    ]

    df = df[feature_cols].dropna()
    return df


# ── Model Building ────────────────────────────────────────────────────────────

def build_preprocessor():
    cat_features = ['batting_team', 'bowling_team', 'city', 'innings_phase']
    num_features = [
        'runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr',
        'pressure_index', 'wicket_resource', 'rr_momentum',
        'balls_wickets', 'venue_chase_rate', 'venue_toss_advantage'
    ]
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features)
    ])


def get_xgb_tuned(X_train, y_train, groups_train):
    """Tune XGBoost with RandomizedSearchCV using GroupKFold."""
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 4, 5, 6],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__min_child_weight': [1, 3, 5],
    }

    pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ))
    ])

    gkf = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=20, cv=gkf, scoring='roc_auc',
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train, groups=groups_train)
    logger.info(f"Best XGB params: {search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def build_ensemble(X_train, y_train):
    """Build soft-voting ensemble: LR + RF + XGB."""
    preprocessor = build_preprocessor()

    lr = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    rf = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
    ])
    xgb = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss',
            use_label_encoder=False
        ))
    ])

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_pipeline(pipe, X_test, y_test, X_train, y_train, groups_train) -> dict:
    """Full evaluation with all metrics + GroupKFold CV."""
    predictions = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    # GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        pipe, X_train, y_train,
        cv=gkf, groups=groups_train,
        scoring='accuracy', n_jobs=-1
    )
    roc_cv = cross_val_score(
        pipe, X_train, y_train,
        cv=gkf, groups=groups_train,
        scoring='roc_auc', n_jobs=-1
    )

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions)),
        'recall': float(recall_score(y_test, predictions)),
        'f1': float(f1_score(y_test, predictions)),
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'brier_score': float(brier_score_loss(y_test, proba)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_scores': cv_scores.tolist(),
        'roc_auc_cv_mean': float(roc_cv.mean()),
        'roc_auc_cv_std': float(roc_cv.std()),
    }

    logger.info("=== Improved Model Evaluation ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    return metrics


def train_improved_model(deliveries: pd.DataFrame, matches: pd.DataFrame,
                         model_type: str = 'ensemble') -> tuple:
    """
    Train improved model and return (pipeline, metrics, feature_df).

    model_type: 'ensemble' | 'xgb_tuned' | 'xgb' | 'rf' | 'logistic'
    """
    logger.info("Building feature matrix...")
    df = build_feature_matrix(deliveries, matches)

    feature_cols = [c for c in df.columns if c not in ['result', 'match_id']]
    X = df[feature_cols]
    y = df['result']
    groups = df['match_id']

    # Match-level train/test split to prevent leakage
    unique_matches = df['match_id'].unique()
    np.random.seed(42)
    test_matches = np.random.choice(
        unique_matches, size=int(len(unique_matches) * 0.2), replace=False
    )
    train_mask = ~df['match_id'].isin(test_matches)
    test_mask = df['match_id'].isin(test_matches)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    logger.info(f"Features: {list(feature_cols)}")

    if model_type == 'ensemble':
        logger.info("Training soft-voting ensemble...")
        pipe = build_ensemble(X_train, y_train)
    elif model_type == 'xgb_tuned':
        logger.info("Tuning XGBoost with GroupKFold + RandomizedSearchCV...")
        pipe = get_xgb_tuned(X_train, y_train, groups_train)
        pipe.fit(X_train, y_train)
    else:
        preprocessor = build_preprocessor()
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'xgb': XGBClassifier(n_estimators=200, random_state=42,
                                  eval_metric='logloss', use_label_encoder=False)
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', models.get(model_type, models['xgb']))
        ])
        pipe.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipe, X_test, y_test, X_train, y_train, groups_train)
    return pipe, metrics, df