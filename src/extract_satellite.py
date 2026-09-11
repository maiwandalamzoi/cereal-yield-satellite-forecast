"""
Feature extraction: country-level seasonal NDVI/EVI from MODIS MOD13Q1
(250 m, 16-day composite, available since 2000-02), aggregated over each
country's boundary (USDOS/LSIB_SIMPLE/2017) for a fixed growing-season
window per year.

Honest scope note: a single fixed window (Feb-Jun) is used for every
country rather than a per-country wheat-calendar (winter wheat in Central
Asia heads/matures roughly Mar-Jun; this is approximate, not phenology-
matched -- see README "Known limitations"). Country-level spatial averaging
also blends non-cropland pixels (desert, urban, mountains) into the signal,
diluting it relative to a cropland-masked extraction.
"""
import csv
import os
import time

from dotenv import load_dotenv
load_dotenv()

import ee

GEE_SA = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")

if not (GEE_SA and GEE_KEY):
    raise SystemExit("GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY not set in .env")

ee.Initialize(ee.ServiceAccountCredentials(GEE_SA, key_data=GEE_KEY))
print("GEE initialized OK")

# ISO3 -> USDOS/LSIB_SIMPLE/2017 country name (the LSIB layer uses its own
# name strings, not ISO codes, so this maps between the two).
COUNTRY_NAMES = {
    "AFG": "Afghanistan",
    "PAK": "Pakistan",
    "IRN": "Iran",
    "KAZ": "Kazakhstan",
    "UZB": "Uzbekistan",
    "TJK": "Tajikistan",
    "TUR": "Turkey",
    "UKR": "Ukraine",
    "NLD": "Netherlands",
}

YEAR_MIN, YEAR_MAX = 2001, 2023
SEASON_START_MD = (2, 1)   # Feb 1
SEASON_END_MD = (6, 30)    # Jun 30

MODIS_NDVI = ee.ImageCollection("MODIS/061/MOD13Q1")

_lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")


def country_geom(name):
    return _lsib.filter(ee.Filter.eq("country_na", name)).geometry()


def season_stats(geom, year):
    start = ee.Date.fromYMD(year, *SEASON_START_MD)
    end = ee.Date.fromYMD(year, *SEASON_END_MD)
    coll = MODIS_NDVI.filterDate(start, end).filterBounds(geom)
    ndvi = coll.select("NDVI").map(lambda img: img.multiply(0.0001))
    evi = coll.select("EVI").map(lambda img: img.multiply(0.0001))

    ndvi_mean_img = ndvi.mean()
    ndvi_max_img = ndvi.max()
    evi_mean_img = evi.mean()

    reducer = ee.Reducer.mean()
    scale = 250

    def reduce(img):
        return img.reduceRegion(reducer=reducer, geometry=geom, scale=scale,
                                 maxPixels=1e10, bestEffort=True).get(img.bandNames().get(0))

    return {
        "ndvi_season_mean": reduce(ndvi_mean_img),
        "ndvi_season_max": reduce(ndvi_max_img),
        "evi_season_mean": reduce(evi_mean_img),
    }


def main():
    rows = []
    for iso3, name in COUNTRY_NAMES.items():
        geom = country_geom(name)
        print(f"{iso3} ({name}):")
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            try:
                stats = season_stats(geom, year)
                vals = ee.Dictionary(stats).getInfo()
                row = {
                    "country_iso3": iso3,
                    "year": year,
                    "ndvi_season_mean": vals.get("ndvi_season_mean"),
                    "ndvi_season_max": vals.get("ndvi_season_max"),
                    "evi_season_mean": vals.get("evi_season_mean"),
                }
                rows.append(row)
                print(f"  {year}: ndvi_mean={row['ndvi_season_mean']}")
            except Exception as e:
                print(f"  {year}: FAILED {type(e).__name__}: {e}")
            time.sleep(0.2)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/satellite_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country_iso3", "year", "ndvi_season_mean",
                                            "ndvi_season_max", "evi_season_mean"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
