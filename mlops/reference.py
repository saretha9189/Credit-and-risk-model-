"""Reference (training-time) distributions and the Population Stability Index."""
import numpy as np
import pandas as pd

EPS = 1e-4


def make_bins(s: pd.Series, n=10):
    """Quantile bin edges from training data; None when the column is (near) constant."""
    s = s.dropna()
    if s.nunique() < 2:
        return None
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def proportions(s: pd.Series, edges) -> np.ndarray:
    """Share of rows in each bin, plus a final 'missing' bin."""
    idx = np.digitize(s.dropna().to_numpy(), edges[1:-1])
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    counts = np.append(counts, s.isna().sum())
    return counts / max(len(s), 1)


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    r, c = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference(X: pd.DataFrame, scores: np.ndarray) -> dict:
    feats = {}
    for col in X.columns:
        e = make_bins(X[col])
        if e is not None:
            feats[col] = {"edges": e, "props": proportions(X[col], e),
                          "missing": float(X[col].isna().mean())}
    se = make_bins(pd.Series(scores))
    return {"features": feats,
            "score": {"edges": se, "props": proportions(pd.Series(scores), se)}}


def feature_psi(ref: dict, X: pd.DataFrame) -> pd.Series:
    out = {c: psi(r["props"], proportions(X[c], r["edges"])) for c, r in ref["features"].items() if c in X}
    return pd.Series(out).sort_values(ascending=False)


def score_psi(ref: dict, scores: np.ndarray) -> float:
    s = ref["score"]
    return psi(s["props"], proportions(pd.Series(scores), s["edges"]))


def missing_shift(ref: dict, X: pd.DataFrame) -> pd.Series:
    """Change in missing-value rate (percentage points) versus training: catches broken upstream feeds."""
    d = {c: (X[c].isna().mean() - r["missing"]) * 100 for c, r in ref["features"].items() if c in X}
    return pd.Series(d).sort_values(key=abs, ascending=False)
