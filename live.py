"""Live odds feeds. Set ODDS_API_KEY to use The Odds API; otherwise a simulated drifting market is used."""
import json, os, urllib.request
import numpy as np, pandas as pd

COLS = ["home", "away", "home_ml", "away_ml"]

def dec_to_american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))

class SimulatedFeed:
    """Mean-reverting random walk around a noisy prior of each game's win probability, with 4.5% vig."""
    name = "Simulated market"
    def __init__(self, priors: dict, seed=11):
        rng = np.random.default_rng(seed); self.rng = np.random.default_rng()
        self.prior = priors
        self.p = {g: float(np.clip(pr + rng.normal(0, .05), .2, .85)) for g, pr in priors.items()}
    def snapshot(self):
        rows = []
        for g, pr in self.prior.items():
            self.p[g] = float(np.clip(self.p[g] + .2 * (pr - self.p[g]) + self.rng.normal(0, .012), .2, .85))
            hd, ad = 1 / (self.p[g] * 1.045), 1 / ((1 - self.p[g]) * 1.045)
            rows.append(dict(home=g[0], away=g[1], home_ml=dec_to_american(hd), away_ml=dec_to_american(ad)))
        return pd.DataFrame(rows, columns=COLS)

class OddsAPIFeed:
    """Real moneylines. Team names must match the names your model was trained on."""
    name = "The Odds API (live)"
    def __init__(self, key, teams): self.key, self.teams = key, set(teams)
    def snapshot(self):
        url = ("https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
               f"?regions=us&markets=h2h&oddsFormat=american&apiKey={self.key}")
        data = json.load(urllib.request.urlopen(url, timeout=10))
        rows = []
        for e in data:
            h, a = e["home_team"], e["away_team"]
            if h not in self.teams or a not in self.teams or not e["bookmakers"]: continue
            o = {x["name"]: x["price"] for x in e["bookmakers"][0]["markets"][0]["outcomes"]}
            rows.append(dict(home=h, away=a, home_ml=o[h], away_ml=o[a]))
        return pd.DataFrame(rows, columns=COLS)

def get_feed(B, state):
    if "feed" not in state:
        key, P = os.getenv("ODDS_API_KEY"), B["profiles"]
        if key: state["feed"] = OddsAPIFeed(key, P.index)
        else:
            from features import diff
            teams = list(np.random.default_rng(3).permutation(P.index))
            pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams) - 1, 2)]
            pri = {g: float(B["clf"].predict_proba(diff(P.loc[g[0]], P.loc[g[1]]).to_frame().T)[0, 1]) for g in pairs}
            state["feed"] = SimulatedFeed(pri)
    return state["feed"]
