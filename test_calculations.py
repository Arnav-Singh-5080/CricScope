import unittest
import pandas as pd
import numpy as np

# Helper functions replicating the implementation logic in application.py
def calculate_balls_left_df(over, ball):
    # Replicates pandas training logic
    df = pd.DataFrame({'over': over, 'ball': ball})
    balls_bowled = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)
    return df['balls_left'].tolist()

def calculate_crr_df(current_score, over, ball):
    # Replicates pandas training logic
    df = pd.DataFrame({'current_score': current_score, 'over': over, 'ball': ball})
    overs_bowled = (df['over'] - 1) + (df['ball'] / 6)
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    return df['crr'].tolist()

def calculate_rrr_df(runs_left, balls_left):
    # Replicates pandas training logic
    df = pd.DataFrame({'runs_left': runs_left, 'balls_left': balls_left})
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)
    return df['rrr'].tolist()

def calculate_prediction_inputs(target, score, overs):
    # Replicates streamlit/prediction logic
    runs_left = target - score
    balls_left = max(120 - (overs * 6), 0)
    crr = score / overs if overs > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    return runs_left, balls_left, crr, rrr


class TestCricScopeCalculations(unittest.TestCase):

    def test_normal_innings_states(self):
        # 1st over, 1st ball
        self.assertEqual(calculate_balls_left_df([1], [1]), [119])
        self.assertEqual(calculate_crr_df([1], [1], [1]), [6.0]) # 1 run off 1 ball => CRR = 6.0

        # 10th over, 3rd ball (9.5 overs bowled)
        self.assertEqual(calculate_balls_left_df([10], [3]), [63])
        # 57 runs off 57 balls => CRR = 57 / 9.5 = 6.0
        self.assertEqual(calculate_crr_df([57], [10], [3]), [6.0])

        # RRR logic: 100 runs left, 63 balls left
        self.assertAlmostEqual(calculate_rrr_df([100], [63])[0], 600 / 63, places=4)

    def test_final_over_scenarios(self):
        # 20th over, 1st ball
        self.assertEqual(calculate_balls_left_df([20], [1]), [5])
        
        # 20th over, 6th ball (final ball of innings)
        self.assertEqual(calculate_balls_left_df([20], [6]), [0])
        self.assertEqual(calculate_crr_df([120], [20], [6]), [6.0]) # 120 runs off 20 overs => CRR = 6.0

    def test_balls_left_capped_at_zero(self):
        # 20th over, 7th ball (due to extra delivery)
        self.assertEqual(calculate_balls_left_df([20], [7]), [0])
        # 20th over, 9th ball
        self.assertEqual(calculate_balls_left_df([20], [9]), [0])

    def test_rrr_division_by_zero_handling(self):
        # When balls_left = 0, RRR should be 0.0 (no division by zero or infinity)
        self.assertEqual(calculate_rrr_df([10], [0]), [0.0])

    def test_prediction_inputs_normal(self):
        # 10 overs completed, target 180, score 80
        runs_left, balls_left, crr, rrr = calculate_prediction_inputs(180, 80, 10)
        self.assertEqual(runs_left, 100)
        self.assertEqual(balls_left, 60)
        self.assertEqual(crr, 8.0)
        self.assertEqual(rrr, 10.0)

    def test_prediction_inputs_boundaries(self):
        # 0 overs completed (start of innings)
        runs_left, balls_left, crr, rrr = calculate_prediction_inputs(180, 0, 0)
        self.assertEqual(runs_left, 180)
        self.assertEqual(balls_left, 120)
        self.assertEqual(crr, 0.0)
        self.assertEqual(rrr, 9.0)

        # 20 overs completed (completed innings)
        runs_left, balls_left, crr, rrr = calculate_prediction_inputs(180, 150, 20)
        self.assertEqual(runs_left, 30)
        self.assertEqual(balls_left, 0)
        self.assertEqual(crr, 7.5)
        self.assertEqual(rrr, 0.0)


if __name__ == '__main__':
    unittest.main()
