import unittest
import pandas as pd
import numpy as np

# Helper functions replicating the implementation logic in application.py
def calculate_balls_left_df(match_id, wide_runs, noball_runs):
    # Replicates pandas training logic: only legal deliveries consume a ball
    df = pd.DataFrame({'match_id': match_id, 'wide_runs': wide_runs, 'noball_runs': noball_runs})
    df['is_legal_delivery'] = ((df['wide_runs'] == 0) & (df['noball_runs'] == 0)).astype(int)
    balls_bowled = df.groupby('match_id')['is_legal_delivery'].cumsum()
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)
    return df['balls_left'].tolist()

def calculate_crr_df(current_score, match_id, wide_runs, noball_runs):
    # Replicates pandas training logic
    df = pd.DataFrame({'current_score': current_score, 'match_id': match_id,
                       'wide_runs': wide_runs, 'noball_runs': noball_runs})
    df['is_legal_delivery'] = ((df['wide_runs'] == 0) & (df['noball_runs'] == 0)).astype(int)
    balls_bowled = df.groupby('match_id')['is_legal_delivery'].cumsum()
    overs_bowled = balls_bowled / 6
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
        # 1 legal delivery bowled => 119 balls left
        self.assertEqual(calculate_balls_left_df([1], [0], [0]), [119])
        # 1 run off 1 legal ball => CRR = 6.0
        self.assertEqual(calculate_crr_df([1], [1], [0], [0]), [6.0])

        # 57 legal deliveries bowled (9.3 overs) => 63 balls left
        mid = [1] * 57
        self.assertEqual(calculate_balls_left_df(mid, [0]*57, [0]*57)[-1], 63)
        # 57 runs off 57 legal balls => CRR = 57 / 9.5 = 6.0
        self.assertEqual(calculate_crr_df(list(range(1, 58)), mid, [0]*57, [0]*57)[-1], 6.0)

        # RRR logic: 100 runs left, 63 balls left
        self.assertAlmostEqual(calculate_rrr_df([100], [63])[0], 600 / 63, places=4)

    def test_extras_do_not_consume_balls(self):
        # 3 deliveries: legal, wide, legal => only 2 legal balls bowled => 118 left
        balls_left = calculate_balls_left_df([1, 1, 1], [0, 1, 0], [0, 0, 0])
        self.assertEqual(balls_left, [119, 119, 118])

        # No-ball also doesn't consume a ball
        balls_left = calculate_balls_left_df([1, 1], [0, 0], [0, 1])
        self.assertEqual(balls_left, [119, 119])

    def test_final_over_scenarios(self):
        # 115 legal deliveries bowled => 5 balls left
        n = 115
        self.assertEqual(calculate_balls_left_df([1]*n, [0]*n, [0]*n)[-1], 5)

        # 120 legal deliveries (final ball of innings) => 0 balls left
        n = 120
        self.assertEqual(calculate_balls_left_df([1]*n, [0]*n, [0]*n)[-1], 0)
        # 120 runs off 20 overs => CRR = 6.0
        self.assertEqual(calculate_crr_df(list(range(1, 121)), [1]*n, [0]*n, [0]*n)[-1], 6.0)

    def test_balls_left_capped_at_zero(self):
        # An innings with extras: 122 deliveries but only 120 legal => never negative
        n = 122
        wides = [0]*n
        wides[5] = 1
        wides[60] = 1
        balls_left = calculate_balls_left_df([1]*n, wides, [0]*n)
        self.assertEqual(balls_left[-1], 0)
        self.assertTrue(all(b >= 0 for b in balls_left))

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