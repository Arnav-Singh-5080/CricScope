def classify_momentum_regime(win, lose, crr, rrr, wickets_in_hand, balls_left):
    """Classify the current match momentum regime from probability and rate context."""
    if balls_left <= 0:
        return "Match Complete", "The innings is finished, so no active momentum phase applies."
    if wickets_in_hand <= 0:
        return "Bowling Dominance", "All wickets are gone and the bowling side has the upper hand."

    if win >= 0.75:
        return (
            "Batting Dominance",
            "The batting side is in strong control and is likely to close out the chase unless momentum abruptly shifts."
        )
    if lose >= 0.75:
        return (
            "Bowling Dominance",
            "The bowling side is applying sustained pressure and is in a commanding position."
        )

    if balls_left <= 12:
        if abs(win - lose) <= 0.15:
            return (
                "Clutch Moment",
                "The match is now in a high-pressure final phase where one over can decide the outcome."
            )
        return (
            "Clutch Defense" if lose > win else "Clutch Chase",
            "The final overs are approaching and this is a tense phase for both sides."
        )

    if wickets_in_hand <= 4 and rrr > crr + 1.5 and win < 0.6:
        return (
            "Collapse Risk",
            "The batting side faces a serious collapse risk unless it accelerates and protects the remaining wickets."
        )

    if abs(crr - rrr) >= 1.5:
        if crr > rrr:
            return (
                "Batting Momentum",
                "The batting side is scoring faster than the required rate and is building momentum."
            )
        return (
            "Bowling Pressure",
            "The bowling side has forced the batting side into a higher required run rate and is dictating the contest."
        )

    return (
        "Balanced Contest",
        "The match is evenly poised and momentum can swing quickly in either direction."
    )
