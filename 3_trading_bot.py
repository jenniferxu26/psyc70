"""
3_trading_bot.py

1. Load model, features, and raw prices
2. Create DataFrame for backtesting: stream price & predictors
3. Trading strategy
  - If edge > threshold, long
  - If edge < threshold, short
  - else, flat
4. Run backtest and evaluate
"""

from pathlib import Path
import joblib, numpy as np, pandas as pd
import backtrader as bt

# file paths
MODEL_PATH = Path("models/eeg_btc_clf.joblib")
FEATURE_PATH = Path("data/processed/X_features.csv")
PRICE_PATH = Path("data/raw/btc_1min.csv")

# parameters
# starting the trade with $10,000
START_CASH = 10000

### EDIT PARAMETERS HERE ###
EDGE_TH = 0.03      # trade IF |edge| ≥ 2 %
RISK_FRAC = 0.3    # % of equity to commit per trade

## Load models & data
# (estimator, list[str])
clf, feat_cols = joblib.load(MODEL_PATH)

X_raw = pd.read_csv(FEATURE_PATH, index_col=0, parse_dates=True)
price = (pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True)
           ['Close'].rename('close'))

# from 2_train_model.py
def expand_features(df):
    out = df.copy()
    # d1z = first difference of each z‑score band.
    for c in df.columns:
        if c.startswith("z_"):
            out[f"d1_{c}"] = df[c].diff().fillna(0)

    # interaction bewteen neural activity x RSI
    if {"z_Delta", "z_Theta", "rsi14"}.issubset(df.columns):
        out["delta_rsi"] = df["z_Delta"] * df["rsi14"]
        out["theta_rsi"] = df["z_Theta"] * df["rsi14"]

    # short-long moving average over the last 3 minutes
    if "sma_diff" in df.columns:
        out["mom3"] = df["sma_diff"].diff(3).fillna(0)

    return out

X_full = expand_features(X_raw.copy())
price  = price.reindex(X_full.index)
df = pd.concat([price, X_full], axis=1).dropna().sort_index()


## Backtrader feed
class PandasFeatData(bt.feeds.PandasData):
    # re-use close price = open
    lines = tuple(feat_cols)
    params = (('datetime', None), ('open', 0), ('high', -1),
              ('low', -1), ('close', 0), ('volume', -1), ('openinterest', -1),
              ) + tuple((c, i + 1) for i, c in enumerate(feat_cols))

data_feed = PandasFeatData(dataname=df)


## Trading bot
class EEGStrategy(bt.Strategy):
    params = dict(edge_th=EDGE_TH)

    def __init__(self):
        self.clf = clf
        self.dataclose = self.datas[0].close

    def next(self):
        feats = np.array([getattr(self.datas[0], col)[0] for col in feat_cols]).reshape(1, -1)
        edge  = 2.0 * self.clf.predict_proba(feats)[0, 1] - 1.0  # [-1,+1]

        pos_size = self.getposition().size
        price    = float(self.dataclose[0])

        # flatten if conviction too low
        if abs(edge) < self.p.edge_th:
            if pos_size:
                self.close()
            return

        direction  = 1 if edge > 0 else -1
        target_qty = max(0.001, round((self.broker.getvalue() * RISK_FRAC) / price, 3))

        # already in desired direction = hold
        if np.sign(pos_size) == direction:
            return

        # flip or enter position
        if pos_size:
            self.close()
        (self.buy if direction == 1 else self.sell)(size=target_qty)

## Run backtest
cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.setcash(START_CASH)
cerebro.adddata(data_feed)
cerebro.addstrategy(EEGStrategy)

# Sharpe ratio & draw down
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                    timeframe=bt.TimeFrame.Minutes)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

res = cerebro.run()[0]
final_eq = cerebro.broker.getvalue()
sharpe = res.analyzers.sharpe.get_analysis().get('sharperatio')
max_dd = res.analyzers.dd.get_analysis().max.drawdown

print("===== Results =====")
print(f"Final equity = ${final_eq:,.2f}")
print(f"Sharpe ratio = {sharpe:.2f}")
print(f"Max draw-down = {max_dd:.2f}%")
