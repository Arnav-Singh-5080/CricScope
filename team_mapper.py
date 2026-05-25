TEAM_MAPPING = {
    "Punjab Kings": "Kings XI Punjab",
    "Delhi Capitals": "Delhi Daredevils",
}

GHOST_TEAMS = {
    'Deccan Chargers', 'Gujarat Lions', 'Kochi Tuskers Kerala',
    'Pune Warriors', 'Rising Pune Supergiant', 'Rising Pune Supergiants'
}

def standardize_team_name(team: str) -> str:
    """Convert current team names to training names"""
    return TEAM_MAPPING.get(team.strip(), team.strip())

def clean_teams(df):
    """Remove ghost teams and standardize names"""
    df = df.copy()                    
    for col in ['batting_team', 'bowling_team', 'team1', 'team2', 'winner']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(standardize_team_name)
            df = df[~df[col].isin(GHOST_TEAMS)]
    return df