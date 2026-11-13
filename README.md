# Sports Analytics & Win Prediction

## Structure
```
sports-analytics/
├── app.py            # Streamlit dashboard (4 tabs)
├── requirements.txt
└── src/
    ├── data.py       # synthetic ingest / load_csv / cleaning
    ├── features.py   # leakage-safe rolling features, home-minus-away diffs
    └── model.py      # training, evaluation, permutation importance
```
Run: `pip install -r requirements.txt && streamlit run app.py`

## Using real data
Provide a CSV with columns: date, home, away, home_pts, away_pts, home_tov, away_tov,
home_efg, away_efg, home_inj, away_inj, then call `run_pipeline(load_csv("games.csv"))`.

## Methodology
- **Cleaning**: dedupe, clip score outliers, per-team median imputation.
- **Leakage control**: rolling means use `shift(1)`, so a game's features only see earlier games.
- **Features**: home-minus-away differentials of 10-game scoring, points allowed, turnover rate,
  eFG%, 5-game win rate, rest days, injuries. Home-court advantage is absorbed by the model
  intercept (every row is from the home perspective).
- **Classifier**: Logistic Regression (linear, well-calibrated baseline) vs Gradient Boosting
  (captures interactions); chosen by log-loss, a proper scoring rule that rewards calibrated probabilities.
- **Regressors**: Gradient Boosting on home and away points; the margin is their difference.
- **Validation**: chronological 80/20 split (no shuffling), reported with accuracy, precision,
  recall, F1, ROC-AUC, Brier score, log-loss, plus a majority-class baseline.
- **Importance**: permutation importance on the test set; less biased than impurity importance
  for correlated features, though correlated features can split credit.
