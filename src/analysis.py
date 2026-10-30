"""Turn fitted models into the plain numbers the 3D dashboard draws."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import SEED
from .data import TARGET

AXIS_COLS = ["AGE", "PERSONAL_MONTHLY_INCOME", "MONTHS_IN_RESIDENCE", "PERSONAL_ASSETS_VALUE",
             "QUANT_DEPENDANTS"]


def r(v, n=4):
    return [round(float(x), n) for x in v]


def roc_and_cutoffs(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 90).astype(int))
    roc = {"fpr": r(fpr[keep]), "tpr": r(tpr[keep]), "thr": r(np.clip(thr[keep], 0, 1))}
    y = np.asarray(y); p = np.asarray(p)
    rows = []
    for c in np.round(np.arange(0.05, 0.96, 0.01), 2):
        appr = p < c
        dec = ~appr
        rows.append({
            "cut": float(c),
            "approve": round(float(appr.mean()), 4),
            "bad_in_approved": round(float(y[appr].mean()) if appr.any() else 0, 4),
            "bads_caught": round(float(dec[y == 1].mean()), 4),
            "goods_lost": round(float(dec[y == 0].mean()), 4),
        })
    return roc, rows


def deciles(y, p):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
    g = d.groupby("bin").agg(pd=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return [{"decile": int(i) + 1, "pd": round(float(a.pd), 4), "actual": round(float(a.actual), 4),
             "n": int(a.n)} for i, a in g.iterrows()]


def scatter_sample(raw_te, p_te, n=1800):
    idx = np.random.RandomState(SEED).choice(len(raw_te), size=min(n, len(raw_te)), replace=False)
    s = raw_te.iloc[idx]
    out = {"pd": r(p_te[idx]), "actual": [int(v) for v in s[TARGET]]}
    for c in AXIS_COLS:
        v = pd.to_numeric(s[c], errors="coerce").fillna(0)
        if c in ("PERSONAL_MONTHLY_INCOME", "PERSONAL_ASSETS_VALUE"):
            v = np.log10(1 + v.clip(upper=v.quantile(0.995)))
        out[c] = r(v, 3)
    return out


def risk_surface(model, fb, raw_te, n_rows=300):
    base = raw_te.sample(n_rows, random_state=SEED).copy()
    ages = np.arange(20, 66, 3)
    incomes = np.unique(np.quantile(raw_te["PERSONAL_MONTHLY_INCOME"], np.linspace(0.04, 0.96, 14)).round(-1))
    # MONTHS_IN_THE_JOB is 0 for 99.8% of applicants, so the third dimension is time at address.
    jobs = np.unique(np.quantile(raw_te["MONTHS_IN_RESIDENCE"].dropna(), [0.1, 0.3, 0.5, 0.7, 0.9]).round())
    frames = []
    for j in jobs:
        grid = [(a, i) for i in incomes for a in ages]
        big = pd.concat([base] * len(grid), ignore_index=True)
        big["AGE"] = np.repeat([g[0] for g in grid], len(base))
        big["PERSONAL_MONTHLY_INCOME"] = np.repeat([g[1] for g in grid], len(base))
        big["MONTHS_IN_RESIDENCE"] = j
        pr = model.predict_proba(fb.transform(big))[:, 1].reshape(len(grid), len(base)).mean(axis=1)
        z = pr.reshape(len(incomes), len(ages))
        frames.append({"months": int(j), "z": [r(row) for row in z]})
    return {"age": [int(a) for a in ages], "income": [int(i) for i in incomes], "frames": frames}


def risk_bars(df, rows="PROFESSION_CODE", cols="OCCUPATION_TYPE", top=8, min_n=40):
    d = df.dropna(subset=[rows, cols])
    top_rows = d[rows].value_counts().head(top).index.sort_values()
    col_vals = sorted(d[cols].unique())
    cells = []
    for i, rv in enumerate(top_rows):
        for j, cv in enumerate(col_vals):
            sub = d[(d[rows] == rv) & (d[cols] == cv)]
            if len(sub) >= min_n:
                cells.append({"i": i, "j": j, "rate": round(float(sub[TARGET].mean()), 4), "n": int(len(sub))})
    return {"rows_name": rows, "cols_name": cols, "rows": [int(v) for v in top_rows],
            "cols": [int(v) for v in col_vals], "cells": cells,
            "overall": round(float(df[TARGET].mean()), 4)}


def drivers(gbm_imp, lr_imp, iv, desc, k=10):
    top = gbm_imp.sort_values(ascending=False).head(k).index.tolist()
    def norm(s):
        m = s.max() or 1
        return [round(float(s.get(f, 0) / m), 4) for f in top]
    return {"features": top, "descriptions": [desc.get(f, "") for f in top],
            "gbm": norm(gbm_imp), "lr": norm(lr_imp), "iv": norm(iv),
            "gbm_raw": [round(float(gbm_imp.get(f, 0)), 5) for f in top],
            "lr_raw": [round(float(lr_imp.get(f, 0)), 5) for f in top],
            "iv_raw": [round(float(iv.get(f, 0)), 4) for f in top]}
