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
    # Expansion set -- LSIB country_na strings confirmed against the actual
    # USDOS/LSIB_SIMPLE/2017 collection before use (several, e.g. Russia,
    # China, Canada, United States, have multiple LSIB features for
    # exclaves/islands; filtering by name and taking .geometry() unions them
    # correctly).
    "CHN": "China",
    "IND": "India",
    "USA": "United States",
    "CAN": "Canada",
    "AUS": "Australia",
    "ARG": "Argentina",
    "EGY": "Egypt",
    "MAR": "Morocco",
    "KGZ": "Kyrgyzstan",
    "AZE": "Azerbaijan",
    "GEO": "Georgia",
    "FRA": "France",
    "DEU": "Germany",
    "TKM": "Turkmenistan",
}

YEAR_MIN, YEAR_MAX = 2001, 2023
SEASON_START_MD = (2, 1)   # Feb 1
SEASON_END_MD = (6, 30)    # Jun 30

MODIS_NDVI = ee.ImageCollection("MODIS/061/MOD13Q1")

_lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")


def country_geom(name):
    return _lsib.filter(ee.Filter.eq("country_na", name)).geometry()


MONTHS = [2, 3, 4, 5, 6]  # Feb-Jun, matching SEASON_START_MD/SEASON_END_MD

def season_stats(geom, year):
    """Season aggregate (mean/max over the whole Feb-Jun window) PLUS a
    monthly breakdown (mean NDVI/EVI per calendar month) in the same
    dictionary -- bundled into one server-side ee.Dictionary so this is
    still a single .getInfo() round-trip per country-year, not 5x more
    network calls for 5x more granularity."""
    start = ee.Date.fromYMD(year, *SEASON_START_MD)
    end = ee.Date.fromYMD(year, *SEASON_END_MD)
    coll = MODIS_NDVI.filterDate(start, end).filterBounds(geom)
    ndvi = coll.select("NDVI").map(lambda img: img.multiply(0.0001))
    evi = coll.select("EVI").map(lambda img: img.multiply(0.0001))

    reducer = ee.Reducer.mean()
    scale = 250
    # bestEffort=True below auto-coarsens this for continental-scale countries
    # (USA, China, Canada, Australia) whose pixel count at 250m exceeds
    # maxPixels -- computation still completes, just at a coarser effective
    # resolution for those specific countries. Worth knowing before trusting
    # their NDVI numbers to the same precision as e.g. Netherlands.

    def reduce(img, band):
        return img.reduceRegion(reducer=reducer, geometry=geom, scale=scale,
                                 maxPixels=1e10, bestEffort=True).get(band)

    result = {
        "ndvi_season_mean": reduce(ndvi.mean(), "NDVI"),
        "ndvi_season_max": reduce(ndvi.max(), "NDVI"),
        "evi_season_mean": reduce(evi.mean(), "EVI"),
    }
    for m in MONTHS:
        m_start = ee.Date.fromYMD(year, m, 1)
        m_end = m_start.advance(1, "month")
        m_ndvi = ndvi.filterDate(m_start, m_end).mean()
        m_evi = evi.filterDate(m_start, m_end).mean()
        result[f"ndvi_m{m:02d}"] = reduce(m_ndvi, "NDVI")
        result[f"evi_m{m:02d}"] = reduce(m_evi, "EVI")
    return result


MONTHLY_FIELDS = [f"{v}_m{m:02d}" for m in MONTHS for v in ("ndvi", "evi")]
FIELDNAMES = ["country_iso3", "year", "ndvi_season_mean", "ndvi_season_max",
              "evi_season_mean"] + MONTHLY_FIELDS


def main():
    rows = []
    for iso3, name in COUNTRY_NAMES.items():
        geom = country_geom(name)
        print(f"{iso3} ({name}):")
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            try:
                stats = season_stats(geom, year)
                vals = ee.Dictionary(stats).getInfo()
                row = {"country_iso3": iso3, "year": year}
                row.update({k: vals.get(k) for k in FIELDNAMES if k not in ("country_iso3", "year")})
                rows.append(row)
                print(f"  {year}: ndvi_mean={row['ndvi_season_mean']}")
            except Exception as e:
                print(f"  {year}: FAILED {type(e).__name__}: {e}")
            time.sleep(0.2)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/satellite_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
