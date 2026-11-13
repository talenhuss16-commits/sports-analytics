"""Leakage-safe feature engineering: every rolling stat uses only games BEFORE tipoff."""
import pandas as pd

BASE = ["pts_r", "opp_pts_r", "tov_r", "efg_r", "form5", "rest", "inj"]

def to_long(g: pd.DataFrame) -> pd.DataFrame:
    def side(s, o, home):
        return pd.DataFrame({"game_id": g.index, "date": g.date, "team": g[s], "opp": g[o],
            "is_home": home, "pts": g[f"{s}_pts"], "opp_pts": g[f"{o}_pts"],
            "tov": g[f"{s}_tov"], "efg": g[f"{s}_efg"], "inj": g[f"{s}_inj"]})
    return pd.concat([side("home", "away", 1), side("away", "home", 0)]).sort_values(["team", "date"])

def add_rolling(long: pd.DataFrame, w=10) -> pd.DataFrame:
    long = long.copy()
    long["win"] = (long.pts > long.opp_pts).astype(int)
    grp = long.groupby("team")
    for c in ["pts", "opp_pts", "tov", "efg"]:
        long[f"{c}_r"] = grp[c].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    long["form5"] = grp["win"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    long["rest"] = grp["date"].diff().dt.days.clip(upper=7)
    return long

def diff(H, A):
    """Home-minus-away differentials. Used identically in training and in the simulator."""
    return (H[BASE] - A[BASE]).add_suffix("_diff")

def build_dataset(g: pd.DataFrame):
    long = add_rolling(to_long(g))
    f = long[BASE + ["is_home", "game_id"]]
    H = f[f.is_home == 1].set_index("game_id"); A = f[f.is_home == 0].set_index("game_id")
    X = diff(H, A)
    y = pd.DataFrame({"home_win": (g.home_pts > g.away_pts).astype(int),
                      "home_pts": g.home_pts, "away_pts": g.away_pts})
    ok = X.dropna().index
    return X.loc[ok].sort_index(), y.loc[ok].sort_index(), long

def latest_profile(long: pd.DataFrame) -> pd.DataFrame:
    g = long.sort_values("date").groupby("team")
    p = g.tail(10).groupby("team")[["pts", "opp_pts", "tov", "efg"]].mean().add_suffix("_r")
    p["form5"] = g.tail(5).groupby("team")["win"].mean()
    p["rest"] = 2.0
    p["inj"] = g.tail(1).set_index("team")["inj"]
    return p
