"""
Monte Carlo scenario simulation: instead of one point forecast, produce a
yield *distribution* per country under a named scenario, using the trained
Random Forest model (tied for best on the test set, and simpler to drive
than the fixed-effects model since it needs no country dummy columns).

Design choice, stated plainly: scenarios are built by RESAMPLING this
country's own real historical seasons, stratified by how dry/stressed that
season's NDVI was -- not by inventing independent synthetic shocks to
precip/temp while leaving NDVI untouched. That second approach would be
physically wrong here: the model's own feature importances (see README) put
~95% of its signal on ndvi_season_mean + yield_lag1 and under 2% combined on
precip/temp. A "drought scenario" that only perturbs precip/temp and leaves
NDVI at its historical average would barely move the model's prediction --
not because drought has no effect, but because that construction ignores
that a real drought also suppresses NDVI (it's the downstream, integrated
signal of moisture stress). Resampling real historical (NDVI, EVI, precip,
temp) tuples together preserves however they actually co-moved, with no
invented correlation assumption.

Trade-off, also stated plainly: each country only has ~22 real seasons, so
"drought years" for a country is a resample from as few as ~7 historical
rows (bottom tercile). This is a small-sample stratified bootstrap, not a
parametric model of extreme scenarios -- it can't represent a scenario more
extreme than anything in this country's 2002-2023 record.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "ndvi_season_mean", "ndvi_season_max", "evi_season_mean",
    "season_precip_total_mm", "season_temp_mean_c", "yield_lag1",
]

N_DRAWS = 5000
SCENARIOS = ["driest_tercile", "all_years", "wettest_tercile"]


def load_country_pool(panel, country_iso3, scenario):
    rows = panel[panel["country_iso3"] == country_iso3].copy()
    if scenario != "all_years":
        terciles = rows["ndvi_season_mean"].quantile([1 / 3, 2 / 3]).values
        if scenario == "driest_tercile":
            rows = rows[rows["ndvi_season_mean"] <= terciles[0]]
        elif scenario == "wettest_tercile":
            rows = rows[rows["ndvi_season_mean"] >= terciles[1]]
    return rows


def simulate_country(model, panel, country_iso3, scenario, latest_yield, rng):
    pool = load_country_pool(panel, country_iso3, scenario)
    if len(pool) == 0:
        return None
    draws = pool.sample(n=N_DRAWS, replace=True, random_state=rng.integers(1e9))
    X = draws[["ndvi_season_mean", "ndvi_season_max", "evi_season_mean",
               "season_precip_total_mm", "season_temp_mean_c"]].copy()
    X["yield_lag1"] = latest_yield  # condition on the real, known last-observed yield
    preds = model.predict(X[FEATURES])
    return {
        "country_iso3": country_iso3,
        "scenario": scenario,
        "n_historical_seasons_in_pool": len(pool),
        "p10_kg_ha": np.percentile(preds, 10),
        "p50_kg_ha": np.percentile(preds, 50),
        "p90_kg_ha": np.percentile(preds, 90),
        "mean_kg_ha": preds.mean(),
        "prob_below_last_year": float(np.mean(preds < latest_yield)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model = joblib.load("models/random_forest_yield.joblib")
    panel = pd.read_csv("data/processed/panel.csv")

    results = []
    for country_iso3 in sorted(panel["country_iso3"].unique()):
        latest_row = panel[panel["country_iso3"] == country_iso3].sort_values("year").iloc[-1]
        latest_yield = latest_row["cereal_yield_kg_ha"]
        for scenario in SCENARIOS:
            r = simulate_country(model, panel, country_iso3, scenario, latest_yield, rng)
            if r:
                r["latest_observed_yield_kg_ha"] = latest_yield
                results.append(r)

    out = pd.DataFrame(results)
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/simulation_results.csv", index=False)

    print(f"{'country':5s} {'scenario':16s} {'n_pool':6s} {'P10':>8s} {'P50':>8s} {'P90':>8s} {'P(<last yr)':>12s}")
    for _, r in out.iterrows():
        print(f"{r['country_iso3']:5s} {r['scenario']:16s} {r['n_historical_seasons_in_pool']:<6.0f} "
              f"{r['p10_kg_ha']:8.0f} {r['p50_kg_ha']:8.0f} {r['p90_kg_ha']:8.0f} "
              f"{r['prob_below_last_year']:11.1%}")

    print(f"\nwrote {len(out)} rows -> data/processed/simulation_results.csv")


if __name__ == "__main__":
    main()
