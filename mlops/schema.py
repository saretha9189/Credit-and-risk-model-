"""Input validation for scoring requests (data quality gate in front of the model)."""
import numpy as np
import pandas as pd

# Plausible ranges for the fields we can sanity-check. Anything outside is rejected,
# because a silent unit/scale error upstream is exactly how a model quietly goes wrong.
RANGES = {
    "AGE": (18, 100),
    "PERSONAL_MONTHLY_INCOME": (0, 1e7),
    "PERSONAL_ASSETS_VALUE": (0, 1e9),
    "QUANT_DEPENDANTS": (0, 20),
    "MONTHS_IN_RESIDENCE": (0, 1200),
}
MAX_MISSING_SHARE = 0.5   # reject a record if more than half its model inputs are missing


class ValidationError(ValueError):
    pass


def clean_records(records: list, raw_cols: list) -> pd.DataFrame:
    """Turn JSON records into the same raw layout the model was trained on."""
    if not isinstance(records, list) or not records:
        raise ValidationError("expected a non-empty list of applicant records")
    df = pd.DataFrame(records)
    unknown = [c for c in df.columns if c not in raw_cols]
    df = df.reindex(columns=raw_cols)            # missing fields become NaN, extras are dropped
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].map(lambda v: v.strip().upper() if isinstance(v, str) else v)
    for z in ("RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3"):
        if z in df:
            df[z] = pd.to_numeric(df[z], errors="coerce")
    for col, (lo, hi) in RANGES.items():
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = v.notna() & ((v < lo) | (v > hi))
            if bad.any():
                raise ValidationError(f"{col} out of range [{lo}, {hi}] in record(s) {list(df.index[bad])}")
    share = df.isna().mean(axis=1)
    if (share > MAX_MISSING_SHARE).any():
        raise ValidationError(f"too many missing fields in record(s) {list(df.index[share > MAX_MISSING_SHARE])}")
    df.attrs["unknown_fields"] = unknown
    return df
