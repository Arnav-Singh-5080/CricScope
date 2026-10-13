import unittest

import numpy as np
import pandas as pd

from model_evaluation import (
    calculate_brier_score,
    calculate_phase_wise_metrics,
    calculate_pressure_situation_metrics,
    calculate_season_wise_metrics,
    classify_phase_from_balls_left,
    create_probability_buckets,
    is_pressure_situation,
)


class TestModelEvaluation(unittest.TestCase):
    def test_brier_score(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.2, 0.8, 0.1]

        expected = np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.8 - 1) ** 2, (0.1 - 0) ** 2])

        self.assertAlmostEqual(calculate_brier_score(y_true, y_prob), expected)

    def test_phase_classification(self):
        self.assertEqual(classify_phase_from_balls_left(100), "Powerplay")
        self.assertEqual(classify_phase_from_balls_left(60), "Middle Overs")
        self.assertEqual(classify_phase_from_balls_left(20), "Death Overs")

    def test_pressure_situation_true(self):
        row = pd.Series(
            {
                "rrr": 12,
                "balls_left": 24,
                "wickets": 3,
                "runs_left": 50,
            }
        )

        self.assertTrue(is_pressure_situation(row))

    def test_pressure_situation_false(self):
        row = pd.Series(
            {
                "rrr": 8,
                "balls_left": 50,
                "wickets": 7,
                "runs_left": 30,
            }
        )

        self.assertFalse(is_pressure_situation(row))

    def test_probability_buckets(self):
        predictions = pd.DataFrame(
            {
                "win_probability": [0.05, 0.15, 0.75, 0.85],
                "result": [0, 0, 1, 1],
            }
        )

        buckets = create_probability_buckets(predictions)

        self.assertIn("prediction_bucket", buckets.columns)
        self.assertIn("sample_count", buckets.columns)
        self.assertIn("calibration_gap", buckets.columns)
        self.assertEqual(int(buckets["sample_count"].sum()), 4)

    def test_season_wise_metrics(self):
        predictions = pd.DataFrame(
            {
                "season": [2018, 2018, 2019, 2019],
                "win_probability": [0.8, 0.3, 0.6, 0.4],
                "result": [1, 0, 1, 0],
                "correct_prediction": [1, 1, 1, 1],
            }
        )

        metrics = calculate_season_wise_metrics(predictions)

        self.assertEqual(len(metrics), 2)
        self.assertIn("brier_score", metrics.columns)
        self.assertIn("accuracy", metrics.columns)

    def test_phase_wise_metrics(self):
        predictions = pd.DataFrame(
            {
                "phase": ["Powerplay", "Powerplay", "Death Overs", "Death Overs"],
                "win_probability": [0.7, 0.4, 0.8, 0.2],
                "result": [1, 0, 1, 0],
                "correct_prediction": [1, 1, 1, 1],
            }
        )

        metrics = calculate_phase_wise_metrics(predictions)

        self.assertEqual(len(metrics), 2)
        self.assertIn("calibration_gap", metrics.columns)

    def test_pressure_metrics(self):
        predictions = pd.DataFrame(
            {
                "pressure_situation": [True, True, False],
                "win_probability": [0.25, 0.7, 0.9],
                "result": [0, 1, 1],
                "correct_prediction": [1, 1, 1],
            }
        )

        metrics = calculate_pressure_situation_metrics(predictions)

        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics.iloc[0]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
