"""Expected value (EV) maths. EV% = expected profit per 1 unit staked = p * decimal_odds - 1."""

def american_to_decimal(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)

def devig(dec_a, dec_b):
    """Remove the bookmaker margin by normalising implied probabilities."""
    ia, ib = 1 / dec_a, 1 / dec_b
    return ia / (ia + ib), ib / (ia + ib)

def grade(ev):
    """Letter grade on EV% (per unit staked)."""
    for cut, g in [(8, "A+"), (5, "A"), (3, "B"), (1, "C"), (-3, "D")]:
        if ev >= cut: return g
    return "F"

def kelly(p, dec, fraction=.25):
    """Fractional Kelly stake as % of bankroll (0 when EV <= 0)."""
    return max(0.0, (p * dec - 1) / (dec - 1)) * fraction * 100

def evaluate(home, away, p_model_home, home_ml, away_ml, w=.5):
    """Return two bet rows. Model prob is blended with the no-vig market (w = model weight)."""
    dh, da = american_to_decimal(home_ml), american_to_decimal(away_ml)
    nv_h, nv_a = devig(dh, da)
    p_h = w * p_model_home + (1 - w) * nv_h
    out = []
    for team, p, nv, dec, ml in ((home, p_h, nv_h, dh, home_ml), (away, 1 - p_h, nv_a, da, away_ml)):
        ev = (p * dec - 1) * 100
        out.append({"Game": f"{away} @ {home}", "Bet": team, "Odds": int(ml),
                    "Market %": nv * 100, "Model %": p * 100, "Edge (pts)": (p - nv) * 100,
                    "EV %": ev, "Grade": grade(ev), "Kelly ¼ %": kelly(p, dec)})
    return out
