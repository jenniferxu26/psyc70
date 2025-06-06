"""
# 2_train_model.py

This code trains EEG classifiers top predict the next K-minute Bitcoin price direction.

1. Load features and labels
2. Feature expansion: difference in z-score, EEGxRSI, SMA 3-minute momentum
3. Cross validation
4. Train models: LR, RF, XGBoost, NB
5. Probability calibration
6. Save model
"""

from pathlib import Path
from itertools import product
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# hyper-parameters
K_HORIZON = 5   # next k minutes
TOP_N = 5       # top N most informative features
SEED = 42


## Load data
RAW  = Path("data/raw")
PROC = Path("data/processed")
MODEL= Path("models"); MODEL.mkdir(exist_ok=True)

X0 = pd.read_csv(PROC / "X_features.csv", index_col=0, parse_dates=True)
y0 = pd.read_csv(PROC / "y_labels.csv",  index_col=0, parse_dates=True)["y"]
btc_close = pd.read_csv(RAW / "btc_1min.csv", index_col=0, parse_dates=True)["Close"]


## Helper functions
# risk-adjusted performance = mean (edge * fwd) / std(edge * fwd)
def information_ratio(edge, fwd):
    pnl = edge * fwd
    return pnl.mean() / (pnl.std() + 1e-9)

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

# walk-forward CV: split training/testing
def walk_forward_splits(n, train, test, step):
    idx = np.arange(n)
    s = 0
    while s + train + test <= n:
        yield idx[s : s + train], idx[s + train : s + train + test]
        s += step

# walk-forward CV
def eval_cv(factory, X, y_bin, fwd, train, test, step):
    accs, irs = [], []
    for tr, te in walk_forward_splits(len(X), train, test, step):
        mdl = factory(); mdl.fit(X[tr], y_bin[tr])

        proba     = mdl.predict_proba(X[te])
        # edge = P(up) – P(down) - expected return
        edge      = proba[:, 1] - proba[:, 0]
        # convert to {-1, 1}
        pred_sign = np.where(edge > 0,  1, -1)
        true_sign = np.where(y_bin[te] == 1, 1, -1)

        accs.append(accuracy_score(true_sign, pred_sign))
        irs.append(information_ratio(edge, fwd[te]))
    return np.mean(accs), np.mean(irs)


# combine features
X_exp = expand_features(X0)

# forward log-return
fwd_ret = np.log(btc_close.shift(-K_HORIZON) / btc_close)

# combine features, label, forward return
# [features ..., y, fwd]
df = pd.concat([X_exp, y0.rename("y"), fwd_ret.rename("fwd")], axis=1).dropna()


## Select top-N most informative features
# feature filter, top N largest MI scores
mi       = mutual_info_classif(df[X_exp.columns], df["y"], random_state=SEED)
top_idx  = np.argsort(mi)[-TOP_N:]
top_cols = df[X_exp.columns].columns[top_idx]
feat_sel = []
for col in top_cols:
    if all(abs(df[col].corr(df[c])) < 0.85 for c in feat_sel):\
            feat_sel.append(col)
print(f"Top {TOP_N} features:", list(feat_sel))

X = df[feat_sel].values
fwd = df["fwd"].values
# {-1,+1} ➔ {0,1}
y_bin_np = ((df["y"].astype(int).values + 1) // 2).astype(int)

n_rows = len(df)
print(f"Rows: {n_rows}")

# dataset too small, prevent overfitting
CV_TRAIN, CV_TEST, CV_STEP = 80, 20, 20


## Train models: find the best model
best_ir, best_name, best_factory = -np.inf, None, None

# return accuracy and information ratio
# 1. Logistic Regression
for C in [.05, .1, .2, .5]:
    def lr(C=C):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l1", solver="saga", C=C,
                               max_iter=1000, n_jobs=-1,
                               random_state=SEED, class_weight="balanced"))
    acc, ir = eval_cv(lr, X, y_bin_np, fwd, CV_TRAIN, CV_TEST, CV_STEP)
    print(f"LogReg L1  C={C:<4}  ACC={acc:.3f}  IR={ir:.4f}")
    if ir > best_ir:
        best_ir, best_name, best_factory = ir, f"LR_L1_C{C}", lr

# 2. Random Forest
for n_est, depth in product([200, 400], [None, 10]):
    def rf(n=n_est, d=depth):
        return RandomForestClassifier(
            n_estimators=n, max_depth=d, max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)
    acc, ir = eval_cv(rf, X, y_bin_np, fwd, CV_TRAIN, CV_TEST, CV_STEP)
    print(f"RF n={n_est:<3} d={str(depth):<4}  ACC={acc:.3f}  IR={ir:.4f}")
    if ir > best_ir:
        best_ir, best_name, best_factory = ir, f"RF_{n_est}_{depth}", rf

# 3. XGBoost
def xgb():
    return XGBClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=-1, random_state=SEED)
acc, ir = eval_cv(xgb, X, y_bin_np, fwd, CV_TRAIN, CV_TEST, CV_STEP)
print(f"XGB  ACC={acc:.3f}  IR={ir:.4f}")
if ir > best_ir:
    best_ir, best_name, best_factory = ir, "XGB_shallow", xgb

print(f"\nBest model: {best_name}, IR={best_ir:.4f}")


## Probabilistic calibration (isotonic)
best_raw = best_factory().fit(X, y_bin_np)
calib    = CalibratedClassifierCV(best_raw, cv=3, method="sigmoid")
calib.fit(X, y_bin_np)

joblib.dump((calib, list(feat_sel)), MODEL / "eeg_btc_clf.joblib")
print("Best model saved. Training completed.")
