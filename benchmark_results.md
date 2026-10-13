# Win Probability Model Benchmarking Results

This document contains the evaluation metrics comparing Random Forest and XGBoost classifiers against the baseline Logistic Regression model in CricScope (per issue #617).

All models were trained on the identical features and train-test split (`test_size=0.2`, `random_state=42`) from the CricScope dataset.

## Evaluation Metrics Summary

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (baseline) | 0.8109 | 0.8152 | 0.8342 | 0.8246 | 0.3942 | 0.1284 |
| **Random Forest** | 0.9992 | 0.9992 | 0.9993 | 0.9993 | 0.0315 | 0.0034 |
| **XGBoost** | 0.9963 | 0.9951 | 0.9980 | 0.9965 | 0.0543 | 0.0088 |

## Key Findings

### 1. Performance and Probability Calibration
- **Log Loss**: Lower Log Loss values indicate that the predicted probabilities are much closer to the actual binary outcomes. Both Random Forest (\(0.0315\)) and XGBoost (\(0.0543\)) drastically outperform Logistic Regression (\(0.3942\)).
- **Brier Score**: Brier Score measures the mean squared difference between predicted probabilities and actual outcomes. A Brier score of \(0.0034\) (Random Forest) and \(0.0088\) (XGBoost) compared to \(0.1284\) (Logistic Regression) demonstrates that the tree-based models offer significantly better probability calibration.

### 2. High Accuracy and Data Leakage Warning
> [!NOTE]
> The extremely high accuracy (\(>99\%\)) for Random Forest and XGBoost is due to the **row-level random split** (`train_test_split` with `random_state=42`) used in the baseline pipeline.
> In a ball-by-ball cricket dataset, different balls from the same match end up split between the training and testing sets.
> Since tree-based models are highly expressive, they can easily memorize match-specific patterns (like combinations of teams, cities, targets, and remaining wickets) to predict the outcome of a match using other balls from the same match.
> 
> While this row-level split is kept here to ensure an **apples-to-apples comparison** with the existing Logistic Regression baseline, a match-level split (like `GroupKFold` or custom splitting on unique `match_id`) is highly recommended for production model updates to prevent data leakage.
