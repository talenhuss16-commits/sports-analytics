"""Ingestion + cleaning. Swap generate_games() for load_csv() to use real data."""
import numpy as np, pandas as pd

TEAMS = ["Atlanta","Boston","Chicago","Dallas","Denver","Houston",
         "Miami","Milwaukee","Phoenix","Seattle","Toronto","Utah"]
RAW_COLS = ["date","home","away","home_pts","away_pts","home_tov","away_tov",
            "home_efg","away_efg","home_inj","away_inj"]

def generate_games(n_games=1500, seed=7) -> pd.DataFrame:
    """Synthetic league: latent team strengths drive scores; home edge, injuries, turnovers matter."""
    rng = np.random.default_rng(seed)
    n = len(TEAMS)
    T = pd.DataFrame({"off": rng.normal(113, 3, n), "dfn": rng.normal(113, 3, n),
                      "pace": rng.normal(99, 2, n), "tov": rng.normal(13.5, 1.2, n),
                      "efg": rng.normal(.53, .015, n)}, index=TEAMS)
    dates = pd.date_range("2023-10-20", periods=n_games // 6 + 1)
    rows = []
    for i in range(n_games):
        h, a = rng.choice(TEAMS, 2, replace=False)
        pace = (T.pace[h] + T.pace[a]) / 2
        rec = {"date": dates[i // 6], "home": h, "away": a}
        for side, tm, opp, hca in (("home", h, a, 2.5), ("away", a, h, 0.0)):
            inj = rng.poisson(1.0)
            tov = T.tov[tm] + rng.normal(0, 2)
            efg = T.efg[tm] + rng.normal(0, .03)
            pts = (pace * (T.off[tm] + T.dfn[opp]) / 200 + hca - 1.5 * inj
                   - 0.8 * (tov - T.tov[tm]) + 60 * (efg - T.efg[tm]) + rng.normal(0, 9))
            rec.update({f"{side}_pts": round(pts), f"{side}_tov": tov,
                        f"{side}_efg": efg, f"{side}_inj": inj})
        rows.append(rec)
    df = pd.DataFrame(rows)
    # Simulate dirty data so the cleaning step is meaningful
    for c in ["home_tov", "away_efg", "home_inj"]:
        df.loc[df.sample(frac=.02, random_state=1).index, c] = np.nan
    return pd.concat([df, df.sample(10, random_state=2)], ignore_index=True)

def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])[RAW_COLS]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"])
    for side in ("home", "away"):
        df[f"{side}_pts"] = df[f"{side}_pts"].clip(60, 170)            # outlier guard
        for c in ("tov", "efg", "inj"):                                 # per-team median imputation
            col = f"{side}_{c}"
            df[col] = df[col].fillna(df.groupby(side)[col].transform("median"))
    return df.sort_values("date").reset_index(drop=True)
