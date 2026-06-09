import pandas as pd
import numpy as np


def compute_venue_chase_win_rate(matches_df: pd.DataFrame) -> dict:
    """
    For each venue, % of matches won by the chasing team.
    Chasing team determined by toss_winner + toss_decision.
    """
    df = matches_df.copy()
    df = df[(df['result'] == 'normal') & (df['dl_applied'] == 0)]
    df = df.dropna(subset=['winner', 'venue', 'toss_winner', 'toss_decision'])

    df['chasing_team'] = np.where(
        df['toss_decision'] == 'field',
        df['toss_winner'],
        np.where(df['toss_winner'] == df['team1'], df['team2'], df['team1'])
    )

    df['chase_win'] = (df['winner'] == df['chasing_team']).astype(int)

    stats = df.groupby('venue')['chase_win'].agg(['sum', 'count']).reset_index()
    stats.columns = ['venue', 'chase_wins', 'total']

    global_avg = stats['chase_wins'].sum() / stats['total'].sum()

    stats['rate'] = np.where(
        stats['total'] >= 5,
        stats['chase_wins'] / stats['total'],
        global_avg
    )

    return dict(zip(stats['venue'], stats['rate']))


def compute_toss_win_rate(matches_df: pd.DataFrame) -> dict:
    """
    For each venue, % of matches where toss winner also won the match.
    Captures venue-level toss advantage signal.
    """
    df = matches_df.copy()
    df = df[(df['result'] == 'normal') & (df['dl_applied'] == 0)]
    df = df.dropna(subset=['winner', 'venue', 'toss_winner'])

    df['toss_winner_won'] = (df['toss_winner'] == df['winner']).astype(int)

    stats = df.groupby('venue')['toss_winner_won'].agg(['sum', 'count']).reset_index()
    stats.columns = ['venue', 'toss_wins', 'total']

    global_avg = stats['toss_wins'].sum() / stats['total'].sum()

    stats['rate'] = np.where(
        stats['total'] >= 5,
        stats['toss_wins'] / stats['total'],
        global_avg
    )

    return dict(zip(stats['venue'], stats['rate']))


def compute_team_toss_win_rate(matches_df: pd.DataFrame) -> dict:
    """
    For each team, % of matches where they won the toss AND won the match.
    Pure gut-feel signal — does winning the toss help THIS team specifically?
    UI display only, not fed into model.
    """
    df = matches_df.copy()
    df = df[(df['result'] == 'normal') & (df['dl_applied'] == 0)]
    df = df.dropna(subset=['toss_winner', 'winner'])

    df['toss_and_win'] = (df['toss_winner'] == df['winner']).astype(int)

    stats = df.groupby('toss_winner')['toss_and_win'].agg(['sum', 'count']).reset_index()
    stats.columns = ['team', 'toss_wins', 'total']

    global_avg = stats['toss_wins'].sum() / stats['total'].sum()

    stats['rate'] = np.where(
        stats['total'] >= 5,
        stats['toss_wins'] / stats['total'],
        global_avg
    )

    return dict(zip(stats['team'], stats['rate']))


def get_venue_batting_style(venue_chase_rates: dict) -> dict:
    """
    For each venue, returns whether batting first or chasing
    has historically been more successful.
    chase_rate > 0.5 → chasing favoured
    chase_rate < 0.5 → batting first favoured
    """
    result = {}
    for venue, rate in venue_chase_rates.items():
        if rate > 0.55:
            result[venue] = {
                'style': 'Chase',
                'confidence': 'Strong' if rate > 0.65 else 'Slight',
                'rate': rate
            }
        elif rate < 0.45:
            result[venue] = {
                'style': 'Bat First',
                'confidence': 'Strong' if rate < 0.35 else 'Slight',
                'rate': rate
            }
        else:
            result[venue] = {
                'style': 'Neutral',
                'confidence': 'Toss Up',
                'rate': rate
            }
    return result