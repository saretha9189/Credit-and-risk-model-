"""Loading and cleaning the credit-card application data."""
import pandas as pd

from .config import DATA_FILE, VARIABLE_LIST

TARGET = "TARGET"
NA_TOKENS = ["NULL", "", " "]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, sep="\t", na_values=NA_TOKENS,
                     keep_default_na=False, low_memory=False)
    cols = list(df.columns)
    # The file has two EDUCATION_LEVEL columns; the second is the mate's.
    second = [i for i, c in enumerate(cols) if c in ("EDUCATION_LEVEL.1",)]
    if second:
        cols[second[0]] = "MATE_EDUCATION_LEVEL"
    df.columns = cols
    df = df.rename(columns={"TARGET_LABEL_BAD=1": TARGET})
    for z in ("RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3"):
        df[z] = pd.to_numeric(df[z], errors="coerce")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("object").where(df[c].notna(), None)
        df[c] = df[c].map(lambda v: v.strip().upper() if isinstance(v, str) else v)
    return df


def drop_uninformative(df: pd.DataFrame):
    """Remove the key and any column with a single value."""
    const = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
    drop = ["ID_CLIENT"] + const
    return df.drop(columns=drop), const


def variable_descriptions() -> dict:
    try:
        v = pd.read_csv(VARIABLE_LIST)
        out = {}
        for _, r in v.iterrows():
            out[str(r["Var_Title"])] = str(r["Var_Description"]) if pd.notna(r["Var_Description"]) else ""
        out["MATE_EDUCATION_LEVEL"] = "Mate's educational level"
        return out
    except Exception:
        return {}
