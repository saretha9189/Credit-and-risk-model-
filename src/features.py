"""Feature engineering: one FeatureBuilder used for training and for what-if grids."""
import numpy as np
import pandas as pd

BINARY_YN = ["FLAG_RESIDENCIAL_PHONE", "COMPANY", "FLAG_PROFESSIONAL_PHONE"]
CAT_COLS = ["PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "POSTAL_ADDRESS_TYPE", "SEX",
            "MARITAL_STATUS", "NACIONALITY", "RESIDENCE_TYPE", "OCCUPATION_TYPE",
            "PROFESSION_CODE", "MATE_PROFESSION_CODE", "MATE_EDUCATION_LEVEL", "PRODUCT",
            "RESIDENCIAL_STATE", "PROFESSIONAL_STATE"]
FREQ_COLS = ["STATE_OF_BIRTH", "CITY_OF_BIRTH", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
             "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "RESIDENCIAL_PHONE_AREA_CODE",
             "PROFESSIONAL_PHONE_AREA_CODE"]


class FeatureBuilder:
    def fit(self, df: pd.DataFrame):
        self.cats_ = {c: sorted(df[c].dropna().astype(str).unique()) for c in CAT_COLS if c in df}
        self.freq_ = {c: df[c].astype(str).value_counts(normalize=True).to_dict()
                      for c in FREQ_COLS if c in df}
        self.income_cap_ = df["PERSONAL_MONTHLY_INCOME"].quantile(0.995)
        self.assets_cap_ = df["PERSONAL_ASSETS_VALUE"].quantile(0.995)
        self.raw_cols_ = list(df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        skip = set(CAT_COLS) | set(FREQ_COLS) | set(BINARY_YN)
        for c in df.columns:
            if c in skip or c == "TARGET":
                continue
            X[c] = pd.to_numeric(df[c], errors="coerce")
        for c in BINARY_YN:
            if c in df:
                X[c] = df[c].map({"Y": 1.0, "N": 0.0})
        for c, cats in self.cats_.items():
            X[c] = df[c].astype(str).map({v: float(i) for i, v in enumerate(cats)})
        for c, f in self.freq_.items():
            X[c + "_FREQ"] = df[c].astype(str).map(f).fillna(0.0)

        inc = X["PERSONAL_MONTHLY_INCOME"].clip(upper=self.income_cap_)
        oth = X["OTHER_INCOMES"].clip(lower=0)
        assets = X["PERSONAL_ASSETS_VALUE"].clip(upper=self.assets_cap_)
        total = inc + oth
        X["LOG_INCOME"] = np.log1p(inc)
        X["LOG_TOTAL_INCOME"] = np.log1p(total)
        X["LOG_ASSETS"] = np.log1p(assets)
        X["INCOME_PER_DEPENDANT"] = total / (1 + X["QUANT_DEPENDANTS"])
        X["ASSETS_TO_INCOME"] = assets / (1 + total)
        X["JOB_SHARE_OF_LIFE"] = X["MONTHS_IN_THE_JOB"] / (12 * X["AGE"].clip(lower=18))
        X["NUM_CARDS"] = X[["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS",
                            "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS"]].sum(axis=1)
        return X

    @property
    def cat_names(self):
        return list(self.cats_.keys())
