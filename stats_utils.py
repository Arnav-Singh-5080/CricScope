def calculate_head_to_head_record(matches, team_a, team_b):
    """Calculate head-to-head results for two selected teams."""
    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()

    total = len(h2h)
    team_a_wins = int((h2h["winner"] == team_a).sum())
    team_b_wins = int((h2h["winner"] == team_b).sum())
    no_result = total - team_a_wins - team_b_wins

    return {
        "matches": h2h,
        "total": total,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "no_result": no_result,
        "team_a_pct": round(team_a_wins / total * 100, 1) if total else 0,
        "team_b_pct": round(team_b_wins / total * 100, 1) if total else 0,
    }
