"""
CricScope Win Probability Model Benchmarking Script.
Benchmarks Logistic Regression, Random Forest, and XGBoost on CricScope data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss
)

def main():
    print("Loading data...")
    if not os.path.exists("matches.csv") or not os.path.exists("deliveries.csv"):
        print("Error: matches.csv or deliveries.csv not found in the current working directory.")
        return

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    print("Preprocessing data (replicating application.py pipeline)...")
    # Merge matches and deliveries
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # Total runs in first innings
    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)
    total_df['target'] = total_df['target'] + 1

    df = df.merge(total_df, on='match_id')
    
    # Filter for second innings
    df = df[df['inning'] == 2]

    # Feature calculations
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
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df = final_df.dropna()

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Column preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    # Model definitions
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = []

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)

        # Predict classes and probabilities
        preds = pipe.predict(X_test)
        proba_all = pipe.predict_proba(X_test)
        proba_pos = proba_all[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        logloss = log_loss(y_test, proba_all)
        brier = brier_score_loss(y_test, proba_pos)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Log Loss': logloss,
            'Brier Score': brier
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*80)
    print("                              MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False, formatters={
        'Accuracy': '{:,.4f}'.format,
        'Precision': '{:,.4f}'.format,
        'Recall': '{:,.4f}'.format,
        'F1-Score': '{:,.4f}'.format,
        'Log Loss': '{:,.4f}'.format,
        'Brier Score': '{:,.4f}'.format
    }))
    print("="*80)

    # Save to markdown
    md_content = f"""# Win Probability Model Benchmarking Results

This document contains the evaluation metrics comparing Random Forest and XGBoost classifiers against the baseline Logistic Regression model. All models were trained on the identical features and train-test split (`test_size=0.2`, `random_state=42`) from the CricScope dataset.

## Evaluation Metrics Summary

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for res in results:
        md_content += f"| **{res['Model']}** | {res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['F1-Score']:.4f} | {res['Log Loss']:.4f} | {res['Brier Score']:.4f} |\n"

    md_content += """
## Key Findings

- **Log Loss & Brier Score**: Low log loss and low Brier scores indicate better-calibrated probabilities. Tree-based models typically capture non-linear interactions (e.g., wickets remaining vs. runs required) far better than linear models.
- **Accuracy & F1-Score**: Tree-based models are expected to outperform Logistic Regression in raw prediction performance.
"""

    results_file = "benchmark_results.md"
    with open(results_file, "w") as f:
        f.write(md_content)
    print(f"\nSaved benchmarking results to {results_file}")

if __name__ == "__main__":
    main()
