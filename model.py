"""Training, time-based evaluation, permutation importance."""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix, mean_absolute_error, r2_score)
from data import generate_games, clean
from features import build_dataset, latest_profile

def clf_metrics(y, p):
    yhat = (p >= .5).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}

def run_pipeline(df=None):
    df = clean(generate_games() if df is None else df)
    X, y, long = build_dataset(df)
    k = int(len(X) * .8)                       # chronological split: no future data in training
    tr, te = X.index[:k], X.index[k:]
    cands = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
             "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                    learning_rate=.05, subsample=.8, random_state=0)}
    comp, fitted = {}, {}
    for name, m in cands.items():
        m.fit(X.loc[tr], y.loc[tr, "home_win"])
        comp[name] = clf_metrics(y.loc[te, "home_win"], m.predict_proba(X.loc[te])[:, 1])
        fitted[name] = m
    best = min(comp, key=lambda n: comp[n]["log_loss"])
    clf = fitted[best]
    p = clf.predict_proba(X.loc[te])[:, 1]
    cm = confusion_matrix(y.loc[te, "home_win"], (p >= .5).astype(int))
    imp = permutation_importance(clf, X.loc[te], y.loc[te, "home_win"],
                                 scoring="neg_log_loss", n_repeats=15, random_state=0)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    regs, reg_m = {}, {}
    for t in ("home_pts", "away_pts"):
        r = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.05,
                                      random_state=0).fit(X.loc[tr], y.loc[tr, t])
        pred = r.predict(X.loc[te]); regs[t] = r
        reg_m[t] = {"MAE": mean_absolute_error(y.loc[te, t], pred), "R2": r2_score(y.loc[te, t], pred)}
    return dict(raw=df, long=long, X=X, y=y, clf=clf, best=best, comp=pd.DataFrame(comp).T,
                cm=cm, importance=importance, regs=regs, reg_m=pd.DataFrame(reg_m).T,
                profiles=latest_profile(long), n_train=len(tr), n_test=len(te),
                baseline=max(y.loc[te, "home_win"].mean(), 1 - y.loc[te, "home_win"].mean()))
