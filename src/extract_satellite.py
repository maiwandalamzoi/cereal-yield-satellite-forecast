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

# China/USA/Canada/Australia at the native 250m scale relied on
# bestEffort=True to auto-coarsen -- which in practice meant GEE silently
# retrying at progressively coarser resolutions server-side, taking so long
# per call (multiplied by 13 reduceRegion calls x 23 years) that a full run
# never finished in over 45 minutes. An EXPLICIT coarser scale for just
# these countries is faster and more predictable than letting bestEffort
# negotiate it -- disclosed here, not hidden, since it means these four
# countries' NDVI/EVI numbers are coarser-resolution than e.g. Netherlands.
COARSE_SCALE_COUNTRIES = {"CHN", "USA", "CAN", "AUS", "IND", "ARG"}
# India and Argentina added after the fact: both hit sustained
# "Too many concurrent aggregations" at native 250m immediately after China
# (every single year failed even after 4 retries with backoff -- not a
# transient blip), while similarly-sized Kazakhstan (2.7M km2, vs India's
# 3.3M km2) had completed fine earlier. Most likely explanation: it's not
# purely about one country's own geometry size, it's accumulated quota
# pressure from the immediately preceding heavy coarse-scale country -- so
# anything picking up right after China/USA/CAN/AUS in COUNTRY_NAMES order
# is more exposed than the same country would be in isolation.
COARSE_SCALE_M = 5000  # verified against 2000m for USA: 0.40747 vs 0.40747 -- no
# meaningful difference once you're already averaging over an area this
# heterogeneous, and it's ~2.5x faster per call (8s vs 20s)

def _make_reduce(geom, scale):
    """Returns a reduce(img) -> value closure over one country's geometry
    and scale. Looks up the band name dynamically (img.bandNames().get(0))
    rather than assuming it's still literally "NDVI"/"EVI" -- .multiply()
    doesn't reliably preserve the source band name, and hardcoding it here
    previously caused every single call to fail with "Dictionary does not
    contain key: 'EVI'" (caught in the per-year try/except, so a run
    completed with 0 rows instead of crashing loudly -- worth knowing
    before trusting a script's "0 failures" printout without checking the
    row count it actually wrote).

    A single calendar-month slice of a 16-day MODIS composite can
    legitimately be empty (no composite start date lands in that window
    for that year), which makes .mean()/.max() return a BANDLESS image --
    bandNames() is then an empty list, and .get(0) on an empty list is a
    hard EEException, not a null. has_bands below turns that into an
    honest null value instead of crashing the call over one missing month.
    """
    reducer = ee.Reducer.mean()

    def reduce(img):
        has_bands = img.bandNames().size().gt(0)
        band_name = ee.Algorithms.If(has_bands, img.bandNames().get(0), "none")
        value = img.reduceRegion(reducer=reducer, geometry=geom, scale=scale,
                                  maxPixels=1e10, bestEffort=True).get(band_name)
        return ee.Algorithms.If(has_bands, value, None)
    return reduce


def season_agg_stats(geom, year, scale=250):
    """Just the 3 season-aggregate values, as their own getInfo() call."""
    start = ee.Date.fromYMD(year, *SEASON_START_MD)
    end = ee.Date.fromYMD(year, *SEASON_END_MD)
    coll = MODIS_NDVI.filterDate(start, end).filterBounds(geom)
    ndvi = coll.select("NDVI").map(lambda img: img.multiply(0.0001))
    evi = coll.select("EVI").map(lambda img: img.multiply(0.0001))
    reduce = _make_reduce(geom, scale)
    return {
        "ndvi_season_mean": reduce(ndvi.mean()),
        "ndvi_season_max": reduce(ndvi.max()),
        "evi_season_mean": reduce(evi.mean()),
    }


def month_stats(geom, year, month, scale=250):
    """Just one month's NDVI+EVI mean, as its own getInfo() call. Split out
    from season_agg_stats deliberately: bundling all 13 reduceRegion calls
    (3 season + 10 monthly) into ONE combined ee.Dictionary/getInfo() was
    the actual cause of sustained "Too many concurrent aggregations"
    failures -- not request pacing (cooldowns up to 90s between calls
    didn't fix it) and not individual country size (tiny Egypt failed the
    same way right after a cluster of large countries). GEE evaluates a
    combined expression graph's reduceRegion calls with internal
    concurrency, and a request bundling this many of them can exceed a
    per-request concurrency cap on its own. 6 smaller calls per
    country-year (this function x5 months + season_agg_stats) trades more
    network round-trips for a request shape GEE will actually accept.
    """
    start = ee.Date.fromYMD(year, *SEASON_START_MD)
    end = ee.Date.fromYMD(year, *SEASON_END_MD)
    coll = MODIS_NDVI.filterDate(start, end).filterBounds(geom)
    m_start = ee.Date.fromYMD(year, month, 1)
    m_end = m_start.advance(1, "month")
    # Filter by date on the RAW collection (coll), before .select()+
    # .multiply() -- .multiply() doesn't carry over system:time_start, so
    # filtering an already-multiplied collection (as an earlier version of
    # this did) always returned an empty collection here, silently
    # producing null for every single month.
    m_coll = coll.filterDate(m_start, m_end)
    m_ndvi = m_coll.select("NDVI").map(lambda img: img.multiply(0.0001)).mean()
    m_evi = m_coll.select("EVI").map(lambda img: img.multiply(0.0001)).mean()
    reduce = _make_reduce(geom, scale)
    return {f"ndvi_m{month:02d}": reduce(m_ndvi), f"evi_m{month:02d}": reduce(m_evi)}


MONTHLY_FIELDS = [f"{v}_m{m:02d}" for m in MONTHS for v in ("ndvi", "evi")]
FIELDNAMES = ["country_iso3", "year", "ndvi_season_mean", "ndvi_season_max",
              "evi_season_mean"] + MONTHLY_FIELDS


def _call_with_retry(fn, max_retries=4):
    """GEE throws a transient 'Too many concurrent aggregations' EEException
    under sustained request load -- retry with growing backoff instead of
    treating it as a permanent failure."""
    delay = 3
    for attempt in range(max_retries):
        try:
            return ee.Dictionary(fn()).getInfo()
        except ee.ee_exception.EEException as e:
            if "concurrent" not in str(e).lower() or attempt == max_retries - 1:
                raise
            print(f"    rate-limited, retrying in {delay}s ({attempt + 1}/{max_retries})...", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def already_done(out_path):
    """Countries with a full 23 rows already in the output file -- resume
    support, so a rerun after a stop doesn't redo (and re-bill) work
    that's already safely on disk."""
    if not os.path.exists(out_path):
        return set()
    counts = {}
    with open(out_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["country_iso3"]] = counts.get(row["country_iso3"], 0) + 1
    n_years = YEAR_MAX - YEAR_MIN + 1
    return {iso for iso, n in counts.items() if n >= n_years}


def main():
    out_path = "data/raw/satellite_features.csv"
    os.makedirs("data/raw", exist_ok=True)

    done = already_done(out_path)
    if done:
        print(f"resuming -- already complete: {sorted(done)}", flush=True)
        if done & COARSE_SCALE_COUNTRIES:
            # Egypt (small, native 250m) still failed every single year right
            # after this run's coarse-country cluster despite a 15s
            # per-country cooldown -- the quota pressure is cumulative
            # across the whole run, not per-country, and needs longer than
            # 15s to actually drain. A real, not cosmetic, pause here.
            print("just finished coarse-scale countries -- waiting 90s for GEE's "
                  "concurrent-aggregation quota to actually drain before resuming...", flush=True)
            time.sleep(90)

    # Write incrementally (flush after every country), not just once at the
    # end -- a run over 23 countries with monthly granularity can take a
    # long time, and an all-at-the-end write means a run that gets 90%
    # through and is interrupted (or hits a rate limit) saves *nothing*.
    # Append mode + resume-skip above means a stopped/rerun script picks up
    # where it left off instead of redoing already-saved countries.
    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        w.writeheader()
    total_rows = 0

    for iso3, name in COUNTRY_NAMES.items():
        if iso3 in done:
            continue
        geom = country_geom(name)
        scale = COARSE_SCALE_M if iso3 in COARSE_SCALE_COUNTRIES else 250
        note = f" [coarse {scale}m]" if scale != 250 else ""
        print(f"{iso3} ({name}){note}:", flush=True)
        t0 = time.time()
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            try:
                row = {"country_iso3": iso3, "year": year}
                row.update(_call_with_retry(lambda: season_agg_stats(geom, year, scale)))
                time.sleep(0.5)
                for m in MONTHS:
                    row.update(_call_with_retry(lambda m=m: month_stats(geom, year, m, scale)))
                    time.sleep(0.5)
                w.writerow(row)
                total_rows += 1
                print(f"  {year}: ndvi_mean={row['ndvi_season_mean']}", flush=True)
            except Exception as e:
                print(f"  {year}: FAILED {type(e).__name__}: {e}", flush=True)
            time.sleep(0.5)
        f.flush()
        print(f"  ({time.time() - t0:.0f}s for {iso3})", flush=True)
        if scale != 250:
            print("  cooldown 20s before next country (just did a heavy coarse-scale one)...", flush=True)
            time.sleep(20)

    f.close()
    print(f"\nwrote {total_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
