"""Run the full credit-risk pipeline and write the 3D dashboard.

    python main.py            # train, evaluate, write outputs/
    python main.py --serve    # same, then open the dashboard at http://localhost:8000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src import analysis
from src.config import OUT_DIR, SEED, TEST_SIZE
from src.data import TARGET, drop_uninformative, load_raw, variable_descriptions
from src.features import FeatureBuilder
from src.model import (build_gbm, build_logit, information_value, metrics, perm_importance)
from src.viz3d import write_dashboard


def run():
    OUT_DIR.mkdir(exist_ok=True)
    raw_full = load_raw()
    df, constant = drop_uninformative(raw_full)
    print(f"{len(df):,} applicants, {df.shape[1] - 1} usable variables, bad rate {df[TARGET].mean():.1%}")
    print("dropped constant columns:", ", ".join(constant))

    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    fb = FeatureBuilder().fit(train)
    Xtr, Xte = fb.transform(train), fb.transform(test)
    ytr, yte = train[TARGET], test[TARGET]

    gbm = build_gbm(fb.cat_names, list(Xtr.columns)).fit(Xtr, ytr)
    lr = build_logit(fb.cat_names, list(Xtr.columns)).fit(Xtr, ytr)
    p_g, p_l = gbm.predict_proba(Xte)[:, 1], lr.predict_proba(Xte)[:, 1]
    m_g, m_l = metrics(yte, p_g), metrics(yte, p_l)
    print("gradient boosting", {k: round(v, 3) for k, v in m_g.items()})
    print("logistic scorecard", {k: round(v, 3) for k, v in m_l.items()})

    imp_g = perm_importance(gbm, Xte, yte)
    imp_l = perm_importance(lr, Xte, yte)
    iv = information_value(Xtr, ytr, fb.cat_names)
    desc = variable_descriptions()

    roc, cuts = analysis.roc_and_cutoffs(yte, p_g)
    payload = {
        "n": int(len(df)), "n_features": int(Xtr.shape[1]), "bad_rate": round(float(df[TARGET].mean()), 4),
        "n_train": int(len(train)), "n_test": int(len(test)), "dropped": constant,
        "gbm": m_g, "lr": m_l, "roc": roc, "cutoffs": cuts,
        "deciles": analysis.deciles(yte, p_g),
        "scatter": analysis.scatter_sample(test, p_g),
        "surface": analysis.risk_surface(gbm, fb, test),
        "bars": analysis.risk_bars(df),
        "drivers": analysis.drivers(imp_g, imp_l, iv, desc),
    }
    (OUT_DIR / "model_results.json").write_text(json.dumps(payload))
    write_dashboard(payload, OUT_DIR)
    print("wrote", OUT_DIR / "dashboard.html")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    run()
    if a.serve:
        import functools, http.server, socketserver, webbrowser
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT_DIR))
        with socketserver.TCPServer(("", 8000), h) as s:
            webbrowser.open("http://localhost:8000/dashboard.html")
            print("serving on http://localhost:8000/dashboard.html  (Ctrl+C to stop)")
            s.serve_forever()
