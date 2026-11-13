import os, pandas as pd, streamlit as st
from ev import prop_rows
from live import fetch_props

STAT = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST"}

def render(B, feed):
    key, logs = os.getenv("ODDS_API_KEY"), B.get("logs")
    if not key or logs is None or logs.empty:
        st.info("Player props need live mode: add `ODDS_API_KEY` in Streamlit Secrets (and the app must reach ESPN)."); return
    st.caption("Projection = player's last 10 games (mean and spread) from real box scores. Each prop costs Odds API credits.")
    c1, c2 = st.columns(2)
    n = c1.slider("Games to load props for", 1, 8, 3)
    w = c2.slider("Model weight vs market ", 0.0, 1.0, 0.4, .05)
    if st.button("Load / refresh player props"):
        raw = []
        for g in feed.snapshot().head(n).itertuples(): raw += fetch_props(key, g.id, g.home, g.away)
        st.session_state["props"] = raw
    raw = st.session_state.get("props")
    if not raw: st.write("Press the button to load today's props."); return
    p = pd.DataFrame(raw).pivot_table(index=["game", "player", "market", "line"], columns="side", values="price", aggfunc="first").dropna().reset_index()
    rec = logs.sort_values("date").groupby("player").tail(10)
    agg = rec.groupby("player")[["PTS", "REB", "AST"]].agg(["mean", "std", "count"])
    rows = []
    for r in p.itertuples():
        if r.player not in agg.index: continue
        s = STAT[r.market]; m, sd, k = agg.loc[r.player, (s, "mean")], agg.loc[r.player, (s, "std")], agg.loc[r.player, (s, "count")]
        if k < 4 or pd.isna(sd): continue
        rows += prop_rows(r.game, r.player, s, r.line, r.Over, r.Under, m, max(sd, 1.0), w)
    if not rows: st.warning("No props matched players with enough recent games."); return
    df = pd.DataFrame(rows).sort_values("EV %", ascending=False)
    pos = (df["EV %"] > 0).mean() * 100
    a, b, c = st.columns(3)
    a.metric("+EV props", f"{pos:.0f}%"); b.metric("−EV props", f"{100 - pos:.0f}%"); c.metric("Best EV", f"{df['EV %'].iloc[0]:+.1f}%", f"Grade {df['Grade'].iloc[0]}")
    col = lambda s: ["color:#16a34a;font-weight:600" if v > 0 else "color:#dc2626" for v in s]
    st.dataframe(df.style.apply(col, subset=["EV %"]).format({"Line": "{:.1f}", "Proj": "{:.1f}", "Market %": "{:.1f}", "Model %": "{:.1f}", "EV %": "{:+.2f}", "Odds": "{:+d}"}),
                 use_container_width=True, hide_index=True)
