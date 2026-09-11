"""
Feature extraction: growing-season weather from NASA POWER's daily point API
(no key required), for one representative grain-belt coordinate per country
(not a country-wide average -- NASA POWER is a point API, not an area API;
see README "Known limitations"). Same Feb-Jun season window as
extract_satellite.py, aggregated to total precipitation and mean temperature.
"""
import csv
import os
import time
from datetime import date

import requests

_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
_PARAMS = "T2M,PRECTOTCORR"

# One representative point per country's main grain-growing region
# (not the capital, except where capital and grain belt roughly coincide).
GRAIN_BELT_POINT = {
    "AFG": (36.70, 67.11, "Balkh (Mazar-i-Sharif)"),
    "PAK": (31.42, 73.09, "Punjab (Faisalabad)"),
    "IRN": (32.65, 51.67, "Isfahan"),
    "KAZ": (53.21, 63.62, "Kostanay"),
    "UZB": (39.65, 66.96, "Samarkand"),
    "TJK": (40.28, 69.62, "Khujand"),
    "TUR": (37.87, 32.48, "Konya"),
    "UKR": (49.59, 34.55, "Poltava"),
    "NLD": (52.50, 5.75, "Flevoland"),
}

YEAR_MIN, YEAR_MAX = 2001, 2023
SEASON_START_MD = (2, 1)
SEASON_END_MD = (6, 30)


def fetch_season(lat, lon, year, timeout=30):
    start = date(year, *SEASON_START_MD).strftime("%Y%m%d")
    end = date(year, *SEASON_END_MD).strftime("%Y%m%d")
    resp = requests.get(_BASE, params={
        "parameters": _PARAMS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    params = data.get("properties", {}).get("parameter", {})
    t2m = params.get("T2M", {})
    precip = params.get("PRECTOTCORR", {})
    temps = [v for v in t2m.values() if v is not None and v > -900]
    precips = [v for v in precip.values() if v is not None and v > -900]
    if not temps or not precips:
        return None
    return {
        "season_precip_total_mm": round(sum(precips), 1),
        "season_temp_mean_c": round(sum(temps) / len(temps), 2),
    }


def main():
    rows = []
    for iso3, (lat, lon, region) in GRAIN_BELT_POINT.items():
        print(f"{iso3} ({region} @ {lat},{lon}):")
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            try:
                stats = fetch_season(lat, lon, year)
                if stats is None:
                    print(f"  {year}: no data")
                    continue
                row = {"country_iso3": iso3, "year": year, **stats}
                rows.append(row)
                print(f"  {year}: precip={row['season_precip_total_mm']}mm "
                      f"temp={row['season_temp_mean_c']}C")
            except Exception as e:
                print(f"  {year}: FAILED {type(e).__name__}: {e}")
            time.sleep(0.3)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/weather_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country_iso3", "year",
                                            "season_precip_total_mm", "season_temp_mean_c"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
