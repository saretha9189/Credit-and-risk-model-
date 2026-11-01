"""Train the production candidate and register it with a model card and reference distributions.

    python -m mlops.train_and_register
"""
import hashlib
import json
import time

import joblib
from sklearn.model_selection import train_test_split

from src.config import ARTIFACT_DIR, DATA_FILE, SEED, TEST_SIZE
from src.data import TARGET, drop_uninformative, load_raw
from src.features import FeatureBuilder
from src.model import build_gbm, metrics
from mlops.reference import build_reference

ARTIFACTS = ARTIFACT_DIR
APPROVE_CUTOFF = 0.30   # decline above this predicted probability of default (policy choice, tune in the dashboard)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df, _ = drop_uninformative(load_raw())
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    fb = FeatureBuilder().fit(train)                 # fitted on training data only (no leakage)
    Xtr, Xte = fb.transform(train), fb.transform(test)
    model = build_gbm(fb.cat_names, list(Xtr.columns)).fit(Xtr, train[TARGET])
    m = metrics(test[TARGET], model.predict_proba(Xte)[:, 1])

    # Reference distributions come from OUT-of-sample validation data, not the training rows: the frequency-
    # encoded features see categories the model has already memorised in training, so a training-based
    # reference raised false drift alarms on clean data (PSI > 1 on RESIDENCIAL_BOROUGH_FREQ).
    ref_df, live_df = train_test_split(test, test_size=0.8, stratify=test[TARGET], random_state=SEED)
    Xref = fb.transform(ref_df)
    ref = build_reference(Xref, model.predict_proba(Xref)[:, 1])
    version = time.strftime("%Y%m%d-%H%M%S")
    bundle = {"model": model, "fb": fb, "reference": ref, "raw_cols": [c for c in train.columns if c != TARGET],
              "version": version, "cutoff": APPROVE_CUTOFF}
    joblib.dump(bundle, ARTIFACTS / "model_bundle.joblib")
    live_df.to_pickle(ARTIFACTS / "holdout.pkl")    # unseen data, used to simulate production traffic

    card = {
        "model_version": version, "algorithm": "HistGradientBoostingClassifier",
        "training_rows": int(len(train)), "holdout_rows": int(len(test)), "reference_rows": int(len(ref_df)), "simulated_traffic_rows": int(len(live_df)),
        "training_data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()[:16],
        "holdout_metrics": {k: round(v, 4) for k, v in m.items()},
        "decision_cutoff": APPROVE_CUTOFF,
        "intended_use": "Rank credit card applicants by default risk (coursework / portfolio demonstration).",
        "known_limitations": ["No time dimension in the data, so production drift is simulated",
                              "Moderate discrimination (AUC ~0.65): suitable as a risk ranking, not a sole decision-maker",
                              "No fairness or adverse-impact analysis yet"],
        "monitoring": {"score_psi": "warn >0.10, critical >0.25", "feature_psi": "warn >0.10, critical >0.25",
                       "performance": "warn if AUC falls >0.02 below holdout",
                       "data_quality": "warn if a field's missing rate moves >10 percentage points"},
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
