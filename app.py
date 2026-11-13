import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src.model import run_pipeline
from live_tab import render as render_live
from src.features import diff

st.set_page_config(page_title="Sports Analytics ML", layout="wide")

@st.cache_resource(show_spinner="Training models...")
def load(): return run_pipeline()
B = load(); P = B["profiles"]; teams = list(P.index)

st.title("🏀 Sports Analytics & Win Prediction")
t1, t2, t3, t4, t5 = st.tabs(["Match Simulator", "Team Comparison & Trends", "Feature Importance", "Model Evaluation", "Live EV"])

with t1:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, 0); away = c2.selectbox("Away team", teams, 1)
    if home == away: st.warning("Pick two different teams.")
    else:
        s1, s2 = st.columns(2)
        inj_h = s1.slider(f"{home} key players out", 0, 5, int(P.loc[home, "inj"]))
        inj_a = s2.slider(f"{away} key players out", 0, 5, int(P.loc[away, "inj"]))
        H, A = P.loc[home].copy(), P.loc[away].copy(); H["inj"], A["inj"] = inj_h, inj_a
        x = diff(H, A).to_frame().T
        prob = B["clf"].predict_proba(x)[0, 1]
        ph, pa = B["regs"]["home_pts"].predict(x)[0], B["regs"]["away_pts"].predict(x)[0]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win probability", f"{prob:.1%}")
        m2.metric(f"{away} win probability", f"{1 - prob:.1%}")
        m3.metric("Expected score", f"{ph:.0f} – {pa:.0f}", f"margin {ph - pa:+.1f}")
        st.plotly_chart(px.bar(x=[home, away], y=[ph, pa], labels={"x": "", "y": "Expected points"}),
                        use_container_width=True)

with t2:
    sel = st.multiselect("Teams", teams, teams[:2])
    cols = ["pts_r", "opp_pts_r", "tov_r", "efg_r", "form5"]
    names = ["Scoring", "Points allowed", "Turnovers", "eFG%", "Recent form"]
    if sel:
        mm = (P[cols] - P[cols].min()) / (P[cols].max() - P[cols].min())
        fig = go.Figure()
        for t in sel:
            fig.add_trace(go.Scatterpolar(r=mm.loc[t], theta=names, fill="toself", name=t))
        st.plotly_chart(fig, use_container_width=True)
        L = B["long"][B["long"].team.isin(sel)].dropna(subset=["pts_r"])
        st.plotly_chart(px.line(L, x="date", y="pts_r", color="team",
                        title="Rolling 10-game scoring"), use_container_width=True)
        st.plotly_chart(px.line(L, x="date", y="form5", color="team",
                        title="Rolling 5-game win rate"), use_container_width=True)

with t3:
    st.caption("Permutation importance on held-out games: increase in log-loss when a feature is shuffled.")
    st.plotly_chart(px.bar(B["importance"], orientation="h", labels={"value": "Δ log-loss", "index": ""}),
                    use_container_width=True)

with t4:
    st.write(f"Chronological split: {B['n_train']} train / {B['n_test']} test games. "
             f"Selected model: **{B['best']}** (lowest log-loss). "
             f"Always-pick-majority baseline accuracy: {B['baseline']:.1%}.")
    st.dataframe(B["comp"].style.format("{:.3f}"))
    st.plotly_chart(px.imshow(B["cm"], text_auto=True, x=["Pred away", "Pred home"],
                    y=["Actual away", "Actual home"], title="Confusion matrix"), use_container_width=True)
    st.subheader("Score regressors"); st.dataframe(B["reg_m"].style.format("{:.3f}"))

with t5:
    render_live(B)
