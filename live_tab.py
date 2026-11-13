import pandas as pd, plotly.express as px, streamlit as st
from features import diff
from ev import evaluate
from live import get_feed

def render(B):
    P = B["profiles"]
    c1, c2, c3 = st.columns(3)
    live = c1.toggle("Live updates", True)
    every = c2.select_slider("Refresh every (s)", [3, 5, 10, 30], 5)
    w = c3.slider("Model weight vs market", 0.0, 1.0, 0.5, .05,
                  help="1.0 = trust model fully. Lower is safer: the market is usually well calibrated.")
    feed = get_feed(B, st.session_state)
    st.caption(f"Source: **{feed.name}**. Set `ODDS_API_KEY` for real odds. EV% = expected profit per 1 unit staked.")
    hist = st.session_state.setdefault("ev_hist", [])

    def board():
        snap = feed.snapshot()
        if snap.empty: st.info("No matching games in the feed right now."); return
        rows = []
        for g in snap.itertuples():
            pm = B["clf"].predict_proba(diff(P.loc[g.home], P.loc[g.away]).to_frame().T)[0, 1]
            rows += evaluate(g.home, g.away, pm, g.home_ml, g.away_ml, w)
        df = pd.DataFrame(rows)
        now = pd.Timestamp.now().strftime("%H:%M:%S")
        hist.extend({"time": now, "Bet": f"{r['Bet']} ({r['Game']})", "EV %": r["EV %"]} for r in df.to_dict("records"))
        del hist[:-400]
        pos = (df["EV %"] > 0).mean() * 100
        best = df.loc[df["EV %"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("+EV lines", f"{pos:.0f}%"); m2.metric("−EV lines", f"{100 - pos:.0f}%")
        m3.metric("Best EV", f"{best['EV %']:+.1f}%", f"Grade {best['Grade']}")
        m4.metric("Best bet", best["Bet"], f"{best['Odds']:+d}", delta_color="off")
        col = lambda s: ["color:#16a34a;font-weight:600" if v > 0 else "color:#dc2626" for v in s]
        st.dataframe(df.sort_values("EV %", ascending=False).style
            .apply(col, subset=["EV %", "Edge (pts)"])
            .format({"Market %": "{:.1f}", "Model %": "{:.1f}", "Edge (pts)": "{:+.1f}",
                     "EV %": "{:+.2f}", "Kelly ¼ %": "{:.2f}", "Odds": "{:+d}"}),
            use_container_width=True, hide_index=True)
        st.caption(f"Last update {now}")
        h = pd.DataFrame(hist)
        st.plotly_chart(px.line(h, x="time", y="EV %", color="Bet", title="EV% over time"),
                        use_container_width=True)

    st.fragment(run_every=every if live else None)(board)()
    st.markdown("**Grades (EV% per unit):** A+ ≥ 8 · A ≥ 5 · B ≥ 3 · C ≥ 1 · D ≥ −3 (no real edge) · F < −3. "
                "EV is only as good as the model's probabilities; for entertainment/research, not betting advice.")
