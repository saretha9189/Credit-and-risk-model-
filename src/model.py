"""Train a logistic-regression scorecard benchmark and a gradient-boosting model."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_stat(y, p),
            "brier": float(brier_score_loss(y, p)), "ap": float(average_precision_score(y, p))}


def build_gbm(cat_names, columns):
    mask = [c in cat_names for c in columns]
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_depth=4, min_samples_leaf=60,
        l2_regularization=1.0, categorical_features=mask, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25, random_state=SEED)


def build_logit(cat_names, columns):
    cats = [c for c in columns if c in cat_names]
    nums = [c for c in columns if c not in cat_names]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]), cats),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.05, max_iter=800))])


def perm_importance(model, X, y, n=6000, repeats=3):
    idx = np.random.RandomState(SEED).choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx], y.iloc[idx], scoring="roc_auc",
                               n_repeats=repeats, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).clip(lower=0)


def information_value(X: pd.DataFrame, y: pd.Series, cat_names, bins=10) -> pd.Series:
    out = {}
    good_tot, bad_tot = (y == 0).sum(), (y == 1).sum()
    for c in X.columns:
        s = X[c]
        if s.nunique(dropna=True) < 2:
            continue
        if c in cat_names or s.nunique() <= bins:
            g = s.fillna(-999)
        else:
            try:
                g = pd.qcut(s, bins, duplicates="drop").astype(str).where(s.notna(), "missing")
            except ValueError:
                continue
        t = pd.crosstab(g, y)
        goods = (t.get(0, 0) + 0.5) / good_tot
        bads = (t.get(1, 0) + 0.5) / bad_tot
        out[c] = float(((goods - bads) * np.log(goods / bads)).sum())
    return pd.Series(out)
