"""Real NBA data from ESPN's public JSON endpoints (no key). Cached so refreshes stay cheap."""
import json, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, streamlit as st

SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates="
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event="

def _get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15))

def _n(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def _scoreboard(day):
    out = []
    try: events = _get(SB + day.strftime("%Y%m%d")).get("events", [])
    except Exception: return out
    for e in events:
        if not e["status"]["type"].get("completed"): continue
        c = {x["homeAway"]: x for x in e["competitions"][0]["competitors"]}
        out.append(dict(id=e["id"], date=day, home=c["home"]["team"]["displayName"], away=c["away"]["team"]["displayName"],
                        home_pts=_n(c["home"]["score"]), away_pts=_n(c["away"]["score"])))
    return out

def _players(g):
    rows = []
    try: j = _get(SUM + g["id"])
    except Exception: return rows
    for t in j.get("boxscore", {}).get("players", []):
        for blk in t.get("statistics", []):
            lab = blk.get("labels", [])
            for a in blk.get("athletes", []):
                s = dict(zip(lab, a.get("stats", [])))
                if a.get("didNotPlay") or not _n(s.get("MIN")): continue
                rows.append(dict(date=g["date"], team=t["team"]["displayName"], player=a["athlete"]["displayName"],
                                 PTS=_n(s.get("PTS")), REB=_n(s.get("REB")), AST=_n(s.get("AST"))))
    return rows

@st.cache_data(ttl=3600, show_spinner="Fetching real NBA results...")
def load_real(days=45, player_days=14):
    today = pd.Timestamp.now().normalize()
    with ThreadPoolExecutor(8) as ex:
        games = [g for r in ex.map(_scoreboard, [today - pd.Timedelta(days=i) for i in range(1, days + 1)]) for g in r]
    if len(games) < 60: raise RuntimeError(f"Only {len(games)} games fetched from ESPN")
    df = pd.DataFrame(games)
    # ESPN scoreboard has no turnover/eFG/injury data: constants (zero-variance, ignored by the model)
    for s in ("home", "away"): df[f"{s}_tov"], df[f"{s}_efg"], df[f"{s}_inj"] = 13.5, .53, 0
    recent = [g for g in games if g["date"] >= today - pd.Timedelta(days=player_days)]
    with ThreadPoolExecutor(8) as ex:
        logs = pd.DataFrame([r for rs in ex.map(_players, recent) for r in rs])
    return df.drop(columns="id"), logs

def live_scores():
    rows = []
    for e in _get(SB + pd.Timestamp.now().strftime("%Y%m%d")).get("events", []):
        c = {x["homeAway"]: x for x in e["competitions"][0]["competitors"]}
        rows.append({"Game": f'{c["away"]["team"]["displayName"]} @ {c["home"]["team"]["displayName"]}',
                     "Score": f'{c["away"]["score"]} - {c["home"]["score"]}', "Status": e["status"]["type"]["shortDetail"]})
    return pd.DataFrame(rows)
