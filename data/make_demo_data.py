"""Generate a SYNTHETIC credit-card application dataset with the same columns as the course dataset.

The real course data belongs to the lecturer and is not published. Every value here is random: it is not
derived from, sampled from or fitted to the real applicants. The default label is drawn from a simple
risk score (younger age, lower income, shorter job tenure, more dependants -> higher risk) plus noise,
so the models have something realistic-looking to learn, but nothing in the output says anything about
real people.

    python data/make_demo_data.py            # writes data/demo_data.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N, SEED = 20_000, 7
OUT = Path(__file__).resolve().parent / "demo_data.txt"


def zipf_names(rng, prefix, n_unique, n, a=1.3):
    """High-cardinality text field with a few common values and a long tail (like cities)."""
    ids = np.minimum(rng.zipf(a, n), n_unique)
    return np.array([f"{prefix}{i:04d}" for i in ids])


def generate(n=N, seed=SEED) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    age = np.clip(r.normal(43, 14, n), 18, 85).round().astype(int)
    income = np.exp(r.normal(6.9, 0.55, n)).round(-1) + 300
    other = np.where(r.random(n) < 0.15, np.exp(r.normal(5.5, 0.8, n)).round(), 0)
    assets = np.where(r.random(n) < 0.35, np.exp(r.normal(9.5, 1.3, n)).round(-2), 0)
    dependants = np.minimum(r.poisson(0.9, n), 16)
    job_m = np.where(r.random(n) < 0.7, r.integers(0, 240, n), 0)
    res_m = r.integers(0, 400, n).astype(float)
    cards = {c: (r.random(n) < p).astype(int) for c, p in
             [("FLAG_VISA", .32), ("FLAG_MASTERCARD", .27), ("FLAG_DINERS", .03),
              ("FLAG_AMERICAN_EXPRESS", .02), ("FLAG_OTHER_CARDS", .01)]}
    profession = r.integers(1, 20, n).astype(float)
    occupation = r.integers(1, 7, n).astype(float)
    product = r.integers(1, 4, n)

    # latent risk score -> default label (about 26% bad)
    z = (-0.035 * (age - 43) - 0.55 * (np.log(income) - 6.9) - 0.004 * np.minimum(job_m, 120)
         + 0.10 * dependants - 0.15 * (assets > 0) + 0.18 * (profession % 4 == 0)
         + 0.12 * (product == 3) + r.normal(0, 1.6, n))
    cut = np.quantile(z, 0.74)
    bad = (z > cut).astype(int)

    def sprinkle(a, p):  # random missing values
        a = a.astype(object)
        a[r.random(n) < p] = "NULL"
        return a

    d = {
        "ID_CLIENT": np.arange(1, n + 1), "CLERK_TYPE": "C",
        "PAYMENT_DAY": r.choice([1, 5, 10, 15, 20, 25], n),
        "APPLICATION_SUBMISSION_TYPE": r.choice(["WEB", "CARNET", "0"], n, p=[.55, .1, .35]),
        "QUANT_ADDITIONAL_CARDS": 0, "POSTAL_ADDRESS_TYPE": r.choice([1, 2], n, p=[.95, .05]),
        "SEX": r.choice(["M", "F", "N"], n, p=[.36, .63, .01]),
        "MARITAL_STATUS": r.choice(range(1, 9), n, p=[.45, .25, .08, .06, .06, .04, .03, .03]),
        "QUANT_DEPENDANTS": dependants, "EDUCATION_LEVEL": 0,
        "STATE_OF_BIRTH": zipf_names(r, "ST", 29, n, 1.6), "CITY_OF_BIRTH": zipf_names(r, "CITY", 9000, n, 1.15),
        "NACIONALITY": r.choice([0, 1, 2], n, p=[.01, .98, .01]),
        "RESIDENCIAL_STATE": zipf_names(r, "ST", 27, n, 1.6), "RESIDENCIAL_CITY": zipf_names(r, "CITY", 3500, n, 1.2),
        "RESIDENCIAL_BOROUGH": zipf_names(r, "BOR", 14000, n, 1.15),
        "FLAG_RESIDENCIAL_PHONE": r.choice(["Y", "N"], n, p=[.85, .15]),
        "RESIDENCIAL_PHONE_AREA_CODE": zipf_names(r, "AC", 100, n, 1.5),
        "RESIDENCE_TYPE": sprinkle(r.choice([1, 2, 3, 4, 5, 6], n, p=[.7, .15, .08, .04, .02, .01]), .03),
        "MONTHS_IN_RESIDENCE": sprinkle(res_m, .08), "FLAG_MOBILE_PHONE": "N", "FLAG_EMAIL": r.choice([0, 1], n, p=[.2, .8]),
        "PERSONAL_MONTHLY_INCOME": income, "OTHER_INCOMES": other, **cards,
        "QUANT_BANKING_ACCOUNTS": r.choice([0, 1, 2], n, p=[.7, .29, .01]),
        "QUANT_SPECIAL_BANKING_ACCOUNTS": r.choice([0, 1, 2], n, p=[.7, .29, .01]),
        "PERSONAL_ASSETS_VALUE": assets, "QUANT_CARS": r.choice([0, 1], n, p=[.7, .3]),
        "COMPANY": r.choice(["Y", "N"], n, p=[.2, .8]),
        "PROFESSIONAL_STATE": zipf_names(r, "ST", 28, n, 1.6),
        "PROFESSIONAL_CITY": sprinkle(zipf_names(r, "CITY", 2200, n, 1.2), .68),
        "PROFESSIONAL_BOROUGH": sprinkle(zipf_names(r, "BOR", 5000, n, 1.15), .68),
        "FLAG_PROFESSIONAL_PHONE": r.choice(["Y", "N"], n, p=[.2, .8]),
        "PROFESSIONAL_PHONE_AREA_CODE": zipf_names(r, "AC", 87, n, 1.5),
        "MONTHS_IN_THE_JOB": job_m, "PROFESSION_CODE": sprinkle(profession, .16),
        "OCCUPATION_TYPE": sprinkle(occupation, .15), "MATE_PROFESSION_CODE": sprinkle(r.integers(1, 19, n).astype(float), .58),
        "EDUCATION_LEVEL.1": sprinkle(r.integers(1, 7, n).astype(float), .65),
        "FLAG_HOME_ADDRESS_DOCUMENT": 0, "FLAG_RG": 0, "FLAG_CPF": 1, "FLAG_INCOME_PROOF": 0,
        "PRODUCT": product, "FLAG_ACSP_RECORD": "N", "AGE": age,
        "RESIDENCIAL_ZIP_3": [f"{x:03d}" for x in np.minimum(r.zipf(1.25, n), 794)],
        "PROFESSIONAL_ZIP_3": [f"{x:03d}" for x in np.minimum(r.zipf(1.25, n), 794)],
        "TARGET_LABEL_BAD=1": bad,
    }
    return pd.DataFrame(d)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT} ({len(df):,} rows, bad rate {df['TARGET_LABEL_BAD=1'].mean():.1%})", file=sys.stderr)
