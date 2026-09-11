"""
Merge yield labels + satellite features + weather features into one panel
(country x year), add a lag feature (prior-year yield, same idea as
crop-stress-prediction's stress_lag1), and write data/processed/panel.csv.

Primary target is now wheat_yield_kg_ha (FAOSTAT, wheat-specific) rather
than cereal_yield_kg_ha (World Bank, all-cereal) -- the refinement this
project's own README flagged as a limitation from day one, now that
FAOSTAT's bulk download route works even though their query API is still
down. The World Bank cereal series is kept as a secondary column (not
deleted) so "how much did switching to wheat-specific labels actually
change" stays a real, answerable comparison rather than an assertion.
"""
import pandas as pd


def main():
    cereal_df = pd.read_csv("data/raw/cereal_yield_worldbank.csv")
    wheat_df = pd.read_csv("data/raw/wheat_yield_faostat.csv")
    sat_df = pd.read_csv("data/raw/satellite_features.csv")
    wx_df = pd.read_csv("data/raw/weather_features.csv")

    df = (wheat_df
          .merge(cereal_df[["country_iso3", "year", "cereal_yield_kg_ha"]],
                 on=["country_iso3", "year"], how="left")
          .merge(sat_df, on=["country_iso3", "year"], how="inner")
          .merge(wx_df, on=["country_iso3", "year"], how="inner"))

    df = df.sort_values(["country_iso3", "year"]).reset_index(drop=True)

    # Lag feature: this country's own wheat yield the prior year. Real
    # farmers/analysts have this at prediction time (it's last year's
    # harvest, not a leak of the current year), same justification as
    # stress_lag1 in crop-stress-prediction.
    df["yield_lag1"] = df.groupby("country_iso3")["wheat_yield_kg_ha"].shift(1)

    before = len(df)
    df = df.dropna(subset=["yield_lag1"]).reset_index(drop=True)
    print(f"dropped {before - len(df)} rows with no lag1 (first year per country)")

    out_path = "data/processed/panel.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows, {df['country_iso3'].nunique()} countries, "
          f"years {df['year'].min()}-{df['year'].max()} -> {out_path}")
    print(df[["country_iso3", "year", "wheat_yield_kg_ha", "cereal_yield_kg_ha"]].head())


if __name__ == "__main__":
    main()
