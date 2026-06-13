def generate_explanation(crr, rrr, wickets):

    reasons = []

    if rrr > crr + 2:
        reasons.append("required run rate pressure increasing")

    if crr > rrr:
        reasons.append("batting side has run rate advantage")

    if wickets <= 3:
        reasons.append("low wickets remaining increases pressure")

    if rrr > 12:
        reasons.append("very high chase requirement")

    if not reasons:
        return "Match is evenly balanced with no strong momentum shift."

    return "Win probability is influenced by " + " and ".join(reasons) + "."