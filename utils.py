import pandas as pd
import numpy as np
import logging

TEAM_NAME_MAP = {
    "Chennai Super Kings":       "Chennai Super Kings",
    "CSK":                       "Chennai Super Kings",
    "Delhi Capitals":            "Delhi Capitals",
    "DC":                        "Delhi Capitals",
    "Delhi Daredevils":          "Delhi Capitals",
    "Punjab Kings":              "Punjab Kings",
    "PBKS":                      "Punjab Kings",
    "Kings XI Punjab":           "Punjab Kings",
    "Kolkata Knight Riders":     "Kolkata Knight Riders",
    "KKR":                       "Kolkata Knight Riders",
    "Mumbai Indians":            "Mumbai Indians",
    "MI":                        "Mumbai Indians",
    "Rajasthan Royals":          "Rajasthan Royals",
    "RR":                        "Rajasthan Royals",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "RCB":                       "Royal Challengers Bangalore",
    "Sunrisers Hyderabad":       "Sunrisers Hyderabad",
    "SRH":                       "Sunrisers Hyderabad",
    "Gujarat Titans":            "Gujarat Titans",
    "GT":                        "Gujarat Titans",
    "Lucknow Super Giants":      "Lucknow Super Giants",
    "LSG":                       "Lucknow Super Giants",
}

KNOWN_CITIES = [
    'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Chandigarh', 
    'Kanpur', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 
    'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 
    'Kochi', 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru'
]

def resolve_team_name(raw_name: str) -> str:
    """Map an API/raw team name to the canonical model training name."""
    if not raw_name:
        return raw_name
    if raw_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[raw_name]
    raw_lower = raw_name.lower()
    for key, val in TEAM_NAME_MAP.items():
        if key.lower() in raw_lower or raw_lower in key.lower():
            return val
    return raw_name

def resolve_city_name(venue: str) -> str:
    """Map stadium/venue strings to canonical training city names."""
    if not venue:
        return "Mumbai"
    venue_lower = venue.lower()
    if "bangalore" in venue_lower or "bengaluru" in venue_lower:
        return "Bengaluru"
    if "mumbai" in venue_lower or "wankhede" in venue_lower or "brabourne" in venue_lower or "dy patil" in venue_lower:
        return "Mumbai"
    if "kolkata" in venue_lower or "eden gardens" in venue_lower:
        return "Kolkata"
    if "chennai" in venue_lower or "chidambaram" in venue_lower or "chepauk" in venue_lower:
        return "Chennai"
    if "delhi" in venue_lower or "feroz" in venue_lower or "arun jaitley" in venue_lower:
        return "Delhi"
    if "hyderabad" in venue_lower or "rajiv gandhi" in venue_lower:
        return "Hyderabad"
    if "jaipur" in venue_lower or "sawai mansingh" in venue_lower:
        return "Jaipur"
    if "mohali" in venue_lower or "chandigarh" in venue_lower:
        return "Mohali"
    if "ahmedabad" in venue_lower or "narendra modi" in venue_lower or "motera" in venue_lower:
        return "Ahmedabad"
    if "pune" in venue_lower or "subrata" in venue_lower:
        return "Pune"
    
    for city in KNOWN_CITIES:
        if city.lower() in venue_lower:
            return city
    return "Mumbai"

def parse_overs_to_balls(overs_decimal: float) -> int:
    """Convert decimal overs representation (e.g. 14.2) to total balls bowled."""
    completed_overs = int(overs_decimal)
    balls_in_current_over = int(round((overs_decimal - completed_overs) * 10))
    return (completed_overs * 6) + balls_in_current_over

def calculate_prediction_inputs(target: int, score: int, overs_decimal: float):
    """Calculate runs_left, balls_left, crr, and rrr using canonical equations."""
    runs_left = target - score
    total_balls_bowled = parse_overs_to_balls(overs_decimal)
    balls_left = max(120 - total_balls_bowled, 0)
    
    crr = score / (total_balls_bowled / 6) if total_balls_bowled > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    return runs_left, balls_left, round(crr, 2), round(rrr, 2)
