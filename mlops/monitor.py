"""Production monitoring: drift, data quality and performance, with alert thresholds.

    python -m mlops.monitor        # simulates 6 weeks of traffic and writes outputs/monitoring_report.*

The dataset has no time dimension, so "weeks" are random batches of the held-out set with
documented, injected problems from week 4 on (see SCENARIO). Replace `simulate_weeks` with
batches read from artifacts/prediction_log.jsonl + labelled outcomes to monitor real traffic.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, SEED
from src.data import TARGET
from src.model import metrics
from mlops.reference import feature_psi, missing_shift, score_psi

ARTIFACTS, OUT = ROOT / "artifacts", ROOT / "outputs"
WARN, CRIT = 0.10, 0.25
AUC_DROP_WARN = 0.02  # and > 3 standard errors
MISSING_WARN_PP = 10.0

SCENARIO = {  # what goes wrong, and when
    1: "clean", 2: "clean", 3: "clean",
    4: "incomes fall 15% (cost-of-living squeeze)",
    5: "incomes fall 30% + upstream feed breaks: PROFESSION_CODE 35% missing",
    6: "incomes fall 40% + feed still broken + younger applicant mix",
}


def inject(raw: pd.DataFrame, week: int, rng) -> pd.DataFrame:
    d = raw.copy()
    if week >= 4:
        d["PERSONAL_MONTHLY_INCOME"] = d["PERSONAL_MONTHLY_INCOME"] * {4: 0.85, 5: 0.70, 6: 0.60}[week]
    if week >= 5:
        d.loc[rng.random(len(d)) < 0.35, "PROFESSION_CODE"] = None
    if week >= 6:
        d["AGE"] = (d["AGE"] - 6).clip(lower=18)
    return d


def simulate_weeks(holdout: pd.DataFrame, weeks=6):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(holdout))
    for w, idx in enumerate(np.array_split(order, weeks), start=1):
        yield w, inject(holdout.iloc[idx], w, rng)


def auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC, so small weekly batches don't raise noise alarms."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


def status(v, warn, crit=None):
    if crit is not None and v > crit:
        return "CRITICAL"
    return "WARN" if v > warn else "OK"


def run():
    b = joblib.load(ARTIFACTS / "model_bundle.joblib")
    card = json.loads((ARTIFACTS / "model_card.json").read_text())
    base_auc = card["holdout_metrics"]["auc"]
    holdout = pd.read_pickle(ARTIFACTS / "holdout.pkl")
    rows, heat, alerts = [], {}, []
    for w, batch in simulate_weeks(holdout):
        X = b["fb"].transform(batch)
        p = b["model"].predict_proba(X)[:, 1]
        fp, ms = feature_psi(b["reference"], X), missing_shift(b["reference"], X)
        m = metrics(batch[TARGET], p)
        r = {"week": w, "scenario": SCENARIO[w], "n": len(batch), "score_psi": score_psi(b["reference"], p),
             "max_feature_psi": float(fp.iloc[0]), "worst_feature": fp.index[0],
             "worst_missing_field": ms.index[0], "worst_missing_pp": float(ms.iloc[0]),
             "auc": m["auc"], "gini": m["gini"], "ks": m["ks"], "mean_pd": float(p.mean()),
             "approve_rate": float((p <= b["cutoff"]).mean())}
        r["score_status"] = status(r["score_psi"], WARN, CRIT)
        r["feature_status"] = status(r["max_feature_psi"], WARN, CRIT)
        r["quality_status"] = "WARN" if abs(r["worst_missing_pp"]) > MISSING_WARN_PP else "OK"
        r["auc_se"] = auc_se(r["auc"], int(batch[TARGET].sum()), int((1 - batch[TARGET]).sum()))
        # alert only if the drop is both material (>0.02) and statistically clear (>3 standard errors)
        drop = base_auc - r["auc"]
        r["performance_status"] = "WARN" if drop > max(AUC_DROP_WARN, 3 * r["auc_se"]) else "OK"
        rows.append(r)
        heat[w] = fp.head(8)
        for k, msg in (("score_status", f"score PSI {r['score_psi']:.3f}"),
                       ("feature_status", f"{r['worst_feature']} PSI {r['max_feature_psi']:.3f}"),
                       ("quality_status", f"{r['worst_missing_field']} missing {r['worst_missing_pp']:+.0f}pp"),
                       ("performance_status", f"AUC {r['auc']:.3f} vs {base_auc:.3f} at validation")):
            if r[k] != "OK":
                alerts.append({"week": w, "level": r[k], "check": k.replace("_status", ""), "detail": msg})
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "monitoring_weekly.csv", index=False)
    (OUT / "monitoring_alerts.json").write_text(json.dumps(alerts, indent=2))
    plot(df, heat, base_auc)
    print(df[["week", "score_psi", "max_feature_psi", "worst_feature", "auc", "approve_rate"]].round(3).to_string(index=False))
    print(f"\n{len(alerts)} alerts:")
    for a in alerts:
        print(f"  week {a['week']}  {a['level']:8s} {a['check']:11s} {a['detail']}")
    return df, alerts


def plot(df, heat, base_auc):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(df.week, df.score_psi, marker="o", color="#8e4ec6", label="score PSI")
    ax[0].plot(df.week, df.max_feature_psi, marker="s", color="#f06ab0", label="worst feature PSI")
    ax[0].axhline(WARN, ls="--", c="goldenrod", lw=1); ax[0].axhline(CRIT, ls="--", c="crimson", lw=1)
    ax[0].set(title="Drift vs training (warn 0.10 / critical 0.25)", xlabel="week", ylabel="PSI"); ax[0].legend()
    ax[1].errorbar(df.week, df.auc, yerr=3 * df.auc_se, marker="o", color="#3b1f5c", capsize=3, label="weekly AUC +/- 3 SE")
    ax[1].axhline(base_auc, ls="--", c="grey", lw=1, label="validation AUC")
    ax[1].axhline(base_auc - AUC_DROP_WARN, ls="--", c="goldenrod", lw=1, label="warn level")
    ax[1].set(title="Discrimination (AUC) per week", xlabel="week"); ax[1].legend()
    H = pd.DataFrame(heat).fillna(0)
    im = ax[2].imshow(H.values, aspect="auto", cmap="PuRd", vmin=0, vmax=0.5)
    ax[2].set_xticks(range(H.shape[1]), [f"wk{c}" for c in H.columns]); ax[2].set_yticks(range(len(H)), H.index, fontsize=7)
    ax[2].set_title("Top drifting features (PSI)"); fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(OUT / "monitoring_report.png", dpi=130)


if __name__ == "__main__":
    run()
