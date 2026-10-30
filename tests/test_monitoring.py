import numpy as np
import pandas as pd
import pytest

from mlops.reference import build_reference, feature_psi, missing_shift, psi, score_psi
from mlops.schema import ValidationError, clean_records
from mlops.monitor import auc_se


def frame(n=5000, loc=0.0, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"a": r.normal(loc, 1, n), "b": r.normal(0, 1, n)})


def test_psi_identical_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0)


def test_no_drift_is_below_warn_level():
    ref = build_reference(frame(seed=1), np.random.default_rng(1).random(5000))
    assert feature_psi(ref, frame(seed=2)).max() < 0.10


def test_shifted_feature_is_flagged_and_ranked_first():
    ref = build_reference(frame(seed=1), np.random.default_rng(1).random(5000))
    out = feature_psi(ref, frame(loc=1.0, seed=2))
    assert out.index[0] == "a" and out["a"] > 0.25 and out["b"] < 0.10


def test_missing_shift_detects_broken_feed():
    ref = build_reference(frame(seed=1), np.random.default_rng(1).random(5000))
    cur = frame(seed=2); cur.loc[:1500, "b"] = np.nan
    assert missing_shift(ref, cur).loc["b"] > 25


def test_score_psi_detects_score_shift():
    s = np.random.default_rng(3).random(5000)
    ref = build_reference(frame(), s)
    assert score_psi(ref, s ** 3) > 0.25


def test_auc_se_shrinks_with_sample_size():
    assert auc_se(0.65, 500, 1500) > auc_se(0.65, 5000, 15000)


COLS = ["AGE", "PERSONAL_MONTHLY_INCOME", "SEX", "MARITAL_STATUS"]


def test_schema_rejects_out_of_range_age():
    with pytest.raises(ValidationError):
        clean_records([{"AGE": 5, "PERSONAL_MONTHLY_INCOME": 1000, "SEX": "M", "MARITAL_STATUS": 1}], COLS)


def test_schema_rejects_mostly_empty_record():
    with pytest.raises(ValidationError):
        clean_records([{"AGE": 30}], COLS)


def test_schema_normalises_strings_and_ignores_unknown_fields():
    df = clean_records([{"AGE": 30, "PERSONAL_MONTHLY_INCOME": 900, "SEX": " m ", "MARITAL_STATUS": 2, "FOO": 1}], COLS)
    assert df.loc[0, "SEX"] == "M" and df.attrs["unknown_fields"] == ["FOO"]
