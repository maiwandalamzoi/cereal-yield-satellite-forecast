"""
Ground-truth labels: annual cereal yield (kg/ha) per country, from the World Bank
Open Data API (indicator AG.YLD.CREL.KG). No API key required.

Honest scope note: this is CEREAL yield, not wheat-specific. FAOSTAT has a
wheat-only series (item 15, element 5419) but fenixservices.fao.org was
returning HTTP 521 (server down) as of 2026-09-11 -- this script exists as a
working v1 so the pipeline isn't blocked. Swap in FAOSTAT once it's back for
a wheat-specific refinement (see README "Known limitations").

For the countries in COUNTRIES below, cereal production is wheat-dominated
(FAO puts Afghanistan's cereal area at >80% wheat), so this is a defensible
proxy, not an exact match.
"""
import csv
import time
import requests

_BASE = "https://api.worldbank.org/v2/country/{iso3}/indicator/AG.YLD.CREL.KG"

# Wheat-belt panel + one out-of-regime comparator (Netherlands), echoing the
# multi-country design of crop-stress-prediction.
COUNTRIES = {
    "AFG": "Afghanistan",
    "PAK": "Pakistan",
    "IRN": "Iran, Islamic Rep.",
    "KAZ": "Kazakhstan",
    "UZB": "Uzbekistan",
    "TJK": "Tajikistan",
    "TUR": "Turkiye",
    "UKR": "Ukraine",
    "NLD": "Netherlands",
}

YEAR_MIN, YEAR_MAX = 2001, 2023  # MOD13Q1 (Terra NDVI/EVI) starts 2000-02


def fetch_country(iso3, timeout=25):
    rows = []
    page = 1
    while True:
        resp = requests.get(
            _BASE.format(iso3=iso3),
            params={"format": "json", "per_page": 200, "page": page,
                    "date": f"{YEAR_MIN}:{YEAR_MAX}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2:
            break
        meta, data = payload[0], payload[1] or []
        for row in data:
            if row["value"] is not None:
                rows.append({
                    "country_iso3": iso3,
                    "country_name": COUNTRIES[iso3],
                    "year": int(row["date"]),
                    "cereal_yield_kg_ha": float(row["value"]),
                })
        if page >= meta.get("pages", 1):
            break
        page += 1
    return rows


def main():
    all_rows = []
    for iso3 in COUNTRIES:
        print(f"fetching {iso3} ({COUNTRIES[iso3]}) ...")
        try:
            rows = fetch_country(iso3)
            print(f"  {len(rows)} yearly records")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
        time.sleep(0.5)

    all_rows.sort(key=lambda r: (r["country_iso3"], r["year"]))
    out_path = "data/raw/cereal_yield_worldbank.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country_iso3", "country_name", "year", "cereal_yield_kg_ha"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
