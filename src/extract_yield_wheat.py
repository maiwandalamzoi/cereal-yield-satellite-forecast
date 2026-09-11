"""
Wheat-specific yield labels from FAOSTAT (item "Wheat", element "Yield",
kg/ha) -- the refinement flagged as a known limitation from the start of
this project: the original label source (World Bank AG.YLD.CREL.KG) is
all-cereal, not wheat-specific, because FAOSTAT's own query API
(fenixservices.fao.org) was returning HTTP 521 (server down) when this
repo was first built.

That API is still down as of this script (retried, still 521/timeout).
This uses FAOSTAT's BULK download instead (bulks-faostat.fao.org, a static
file host, unaffected by the query API's outage): the full
Production_Crops_Livestock_E_All_Data_(Normalized).csv (~545 MB
uncompressed, all countries/crops/years), streamed and filtered down to
just Wheat + Yield + our country list + 2001-2023 -- never loaded fully
into memory.

Output: data/raw/wheat_yield_faostat.csv, columns country_iso3, year,
wheat_yield_kg_ha. This becomes the primary training target from here on
(see build_dataset.py / train.py) -- the World Bank cereal-yield series
(extract_yield.py) is kept as a secondary/comparison column, not deleted,
so the "how much did wheat-specific data change the results" question
stays answerable.
"""
import csv
import io
import os
import zipfile

import requests

BULK_URL = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
CSV_NAME = "Production_Crops_Livestock_E_All_Data_(Normalized).csv"

# FAOSTAT's own numeric Area Code -> our ISO3 codes. Matching by code, not
# by name string: FAOSTAT's own bulk CSV export has a broken byte for
# Turkiye's u-with-diaeresis ("T�rkiye" -- neither valid UTF-8 nor
# latin-1 recovers it cleanly), which silently dropped Turkiye entirely
# on a first attempt at name-based matching. Area Codes are stable and
# encoding-proof; confirmed against Production_Crops_Livestock_E_AreaCodes.csv.
AREACODE_TO_ISO3 = {
    "2": "AFG", "165": "PAK", "102": "IRN", "108": "KAZ", "235": "UZB",
    "208": "TJK", "223": "TUR", "230": "UKR", "150": "NLD", "41": "CHN",
    "100": "IND", "231": "USA", "33": "CAN", "10": "AUS", "9": "ARG",
    "59": "EGY", "143": "MAR", "113": "KGZ", "52": "AZE", "73": "GEO",
    "68": "FRA", "79": "DEU", "213": "TKM",
}

YEAR_MIN, YEAR_MAX = 2001, 2023


def download_bulk(dest_path, timeout=180):
    if os.path.exists(dest_path):
        print(f"using cached {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return
    print(f"downloading {BULK_URL} ...")
    resp = requests.get(BULK_URL, timeout=timeout, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"downloaded {os.path.getsize(dest_path):,} bytes")


def extract_wheat_yield(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(CSV_NAME) as fb:
            f = io.TextIOWrapper(fb, encoding="utf-8", errors="replace")
            reader = csv.DictReader(f)
            for row in reader:
                if row["Item"] != "Wheat" or row["Element"] != "Yield":
                    continue
                iso3 = AREACODE_TO_ISO3.get(row["Area Code"])
                if iso3 is None:
                    continue
                year = int(row["Year"])
                if not (YEAR_MIN <= year <= YEAR_MAX):
                    continue
                if row["Unit"] != "kg/ha":
                    print(f"  WARNING skipping {row['Area']} {year}: unexpected unit {row['Unit']!r}")
                    continue
                rows.append({
                    "country_iso3": iso3,
                    "year": year,
                    "wheat_yield_kg_ha": round(float(row["Value"]), 1),
                })
    return rows


def main():
    cache_dir = "data/raw/_cache"
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "faostat_production.zip")
    download_bulk(zip_path)

    rows = extract_wheat_yield(zip_path)
    rows.sort(key=lambda r: (r["country_iso3"], r["year"]))

    found_countries = {r["country_iso3"] for r in rows}
    missing = set(AREACODE_TO_ISO3.values()) - found_countries
    if missing:
        print(f"WARNING: no wheat-yield rows found for: {sorted(missing)}")

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/wheat_yield_faostat.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country_iso3", "year", "wheat_yield_kg_ha"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows, {len(found_countries)} countries -> {out_path}")


if __name__ == "__main__":
    main()
