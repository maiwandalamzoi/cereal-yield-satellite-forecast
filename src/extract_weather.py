"""
Feature extraction: growing-season weather from NASA POWER's daily point API
(no key required), for one representative grain-belt coordinate per country
(not a country-wide average -- NASA POWER is a point API, not an area API;
see README "Known limitations"). Same Feb-Jun season window as
extract_satellite.py: one season aggregate (total precip, mean temp) PLUS a
monthly breakdown, both derived from the SAME single daily-series API call
per country-year -- no extra network requests for the extra granularity,
just finer bucketing of data already fetched.
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
    # Expansion set -- real named wheat/cereal-growing regions, not capitals.
    "CHN": (34.70, 113.60, "Henan"),
    "IND": (30.90, 75.85, "Punjab (Ludhiana)"),
    "USA": (37.70, -97.30, "Kansas (Wichita)"),
    "CAN": (50.45, -104.60, "Saskatchewan (Regina)"),
    "AUS": (-31.48, 118.28, "WA Wheatbelt (Merredin)"),
    "ARG": (-33.90, -60.60, "Pampas (Pergamino)"),
    "EGY": (31.10, 30.90, "Nile Delta (Kafr el-Sheikh)"),
    "MAR": (33.00, -7.60, "Chaouia (Settat)"),
    "KGZ": (42.87, 74.60, "Chuy Valley (Bishkek)"),
    "AZE": (40.68, 46.36, "Aran (Ganja)"),
    "GEO": (41.92, 45.47, "Kakheti (Telavi)"),
    "FRA": (48.45, 1.49, "Beauce (Chartres)"),
    "DEU": (52.13, 11.63, "Magdeburger Borde"),
    "TKM": (37.60, 61.83, "Mary"),
}

YEAR_MIN, YEAR_MAX = 2001, 2023
SEASON_START_MD = (2, 1)
SEASON_END_MD = (6, 30)


MONTHS = [2, 3, 4, 5, 6]  # Feb-Jun, matching SEASON_START_MD/SEASON_END_MD


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
    # keys are "YYYYMMDD" strings -- bucket by month (chars 4:6) for the
    # monthly breakdown, computed from this same daily series, no extra call.
    temps_by_month, precips_by_month = {}, {}
    for k, v in t2m.items():
        if v is not None and v > -900:
            temps_by_month.setdefault(int(k[4:6]), []).append(v)
    for k, v in precip.items():
        if v is not None and v > -900:
            precips_by_month.setdefault(int(k[4:6]), []).append(v)

    all_temps = [v for vs in temps_by_month.values() for v in vs]
    all_precips = [v for vs in precips_by_month.values() for v in vs]
    if not all_temps or not all_precips:
        return None

    result = {
        "season_precip_total_mm": round(sum(all_precips), 1),
        "season_temp_mean_c": round(sum(all_temps) / len(all_temps), 2),
    }
    for m in MONTHS:
        pm, tm = precips_by_month.get(m), temps_by_month.get(m)
        result[f"precip_m{m:02d}_mm"] = round(sum(pm), 1) if pm else None
        result[f"temp_m{m:02d}_c"] = round(sum(tm) / len(tm), 2) if tm else None
    return result


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
    monthly_fields = [f"{v}_m{m:02d}{u}" for m in MONTHS for v, u in (("precip", "_mm"), ("temp", "_c"))]
    fieldnames = ["country_iso3", "year", "season_precip_total_mm", "season_temp_mean_c"] + monthly_fields
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
