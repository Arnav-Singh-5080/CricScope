def resolve_city_name(venue_str: str) -> str:
    """
    Maps raw venue/stadium strings to canonical city names used in the training dataset.
    Prevents OneHotEncoder from creating zero-vectors for unknown categories.
    """
    if not venue_str or not isinstance(venue_str, str):
        return "Unknown"

    # Dictionary mapping famous stadiums (substrings) to their canonical cities
    stadium_to_city = {
        "wankhede": "Mumbai",
        "eden gardens": "Kolkata",
        "sawai mansingh": "Jaipur",
        "chidambaram": "Chennai",
        "chepauk": "Chennai",
        "chinnaswamy": "Bengaluru",
        "arun jaitley": "Delhi",
        "feroz shah kotla": "Delhi",
        "narendra modi": "Ahmedabad",
        "motera": "Ahmedabad",
        "brabourne": "Mumbai",
        "dy patil": "Navi Mumbai",
        "rajiv gandhi": "Hyderabad",
        "punjab cricket association": "Chandigarh",
        "is bindra": "Chandigarh",
        "green park": "Kanpur",
        "holkar": "Indore"
    }

    venue_lower = venue_str.lower()

    # 1. Check if the string contains a known stadium name
    for stadium, city in stadium_to_city.items():
        if stadium in venue_lower:
            return city

    # 2. Fallback to the original logic: if there's a comma, take the last part
    if "," in venue_str:
        return venue_str.split(",")[-1].strip()

    # 3. Default fallback: return the stripped string
    return venue_str.strip()

