# Wheat Yield Forecasting from Satellite + Weather Data

**Live dashboards (click to open):**
- 🌾 **[Results report](https://maiwandalamzoi.github.io/cereal-yield-satellite-forecast/report.html)** — pipeline, model comparison, feature importance, Monte Carlo simulator
- 🗺️ **[Grain Belt Atlas](https://maiwandalamzoi.github.io/cereal-yield-satellite-forecast/atlas.html)** — interactive world map, click any country for its real data sources and risk
- Or start here: **[maiwandalamzoi.github.io/cereal-yield-satellite-forecast](https://maiwandalamzoi.github.io/cereal-yield-satellite-forecast/)**

Predicts a country's annual wheat yield from in-season satellite vegetation
indices and weather, across 23 countries spanning every populated continent —
Afghanistan, Pakistan, Iran, Kazakhstan, Uzbekistan, Tajikistan, Turkiye,
Ukraine, Netherlands, China, India, USA, Canada, Australia, Argentina, Egypt,
Morocco, Kyrgyzstan, Azerbaijan, Georgia, France, Germany, Turkmenistan.

**Dataset:** 506 real ground-truth wheat-yield rows (23 countries × ~22
growing seasons, 2001–2023, FAOSTAT), each joined to satellite (MODIS
NDVI/EVI, monthly Feb–Jun) and weather (NASA POWER, monthly) features — 391
rows for training, 115 held out for testing. See *Method* below for exactly
how each number was produced and *Results* for what it bought.

**What this is:** a real, trained regression model (LightGBM) evaluated on a
genuine train/test split, benchmarked against a naive lag-1 baseline, with
honestly reported metrics.
**What this is not:** a production yield forecaster. See *Known
limitations* below before drawing conclusions from the numbers.

*(Unrelated to this repo's own scope, but for context: a separate project,
[crop-stress-prediction](https://github.com/maiwandalamzoi/crop-stress-prediction),
detects vegetation-stress anomalies 2–4 weeks ahead rather than forecasting
the harvest yield number — different question, same satellite-data
approach.)*

## Method

1. **Labels** — annual wheat yield (kg/ha) per country, from FAOSTAT (item
   "Wheat", element "Yield"), 2001-2023, via their bulk-download route (their
   query API, `fenixservices.fao.org`, was down — HTTP 521 — throughout this
   build). World Bank all-cereal yield (`AG.YLD.CREL.KG`) kept as a secondary
   comparison column, not the target. `src/extract_yield_wheat.py`
   (`src/extract_yield.py` for the cereal comparison series)
2. **Satellite features** — country-level NDVI/EVI from MODIS MOD13Q1
   (250 m/2–5 km, 16-day composite), both a Feb-Jun season aggregate and a
   per-calendar-month breakdown, reduced over each country's boundary via
   Google Earth Engine. `src/extract_satellite.py`
3. **Weather features** — precipitation and mean temperature from NASA
   POWER, both season-total and per-calendar-month, at one representative
   grain-belt coordinate per country (not a country-wide average).
   `src/extract_weather.py`
4. **Panel + lag feature** — merged into one country×year panel with a
   `yield_lag1` feature (that country's own yield the prior year — known at
   prediction time, not a leak). `src/build_dataset.py`
5. **Train/evaluate** — time-based split (train ≤2018, test 2019-2023, no
   random k-fold — a random split would leak future climate patterns into
   training). Six models compared: naive lag1 baseline, Ridge, a country
   fixed-effects linear model, Random Forest, XGBoost, LightGBM.
   `src/train.py`
6. **Scenario simulation** — Monte Carlo drought/heatwave scenarios: resample
   each country's own historical seasons (optionally stratified to represent
   a scenario), run the trained model thousands of times, report a yield
   *distribution* (P10/P50/P90) instead of one number. `src/simulate.py`

## Results

Train: 391 rows (2002–2018). Test: 115 rows (2019–2023), a genuine holdout —
none of these years' weather/satellite/yield were seen during training.

| Model | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---|---|---|---|
| **lightgbm** | **302.3** | **399.1** | **0.962** |
| random forest | 323.9 | 424.2 | 0.957 |
| xgboost | 330.8 | 419.5 | 0.957 |
| fixed-effects (country dummies) | 362.0 | 456.9 | 0.950 |
| ridge | 363.3 | 499.4 | 0.940 |
| baseline (yield = last year) | 351.5 | 498.9 | 0.940 |

**Honest read of this table:** the naive "this year repeats last year"
baseline is still strong (R²=0.940) — most of a country's wheat yield *is*
just its recent level. LightGBM now wins outright (not a tie with another
model), beating the baseline by ~20% RMSE — a bigger, more real lift than
the original 9-country study's ~13%. Worth naming directly: that first
version of this README said LightGBM's stock hyperparameters were "tuned for
far more rows than this study's 153 training rows." With 391 training rows
now, LightGBM goes from the worst model to the best — not tuned differently,
just given enough data, exactly as predicted rather than retroactively
explained.

LightGBM feature importance (top 10 of 22):

| Feature | Importance |
|---|---|
| `yield_lag1` | 0.803 |
| `temp_m06_c` | 0.082 |
| `evi_m06` | 0.042 |
| `ndvi_season_max` | 0.011 |
| `evi_m05` | 0.007 |
| `precip_m04_mm` | 0.005 |
| `evi_m03` | 0.005 |
| `precip_m03_mm` | 0.005 |
| `precip_m05_mm` | 0.004 |
| `precip_m06_mm` | 0.004 |

Monthly resolution changed the story the original season-average version
told: `temp_m06_c` (June temperature — the pre-harvest month) alone carries
8.2%, more than every NDVI/EVI feature combined except season-max. All 10
monthly weather columns together carry ~11% (vs. under 2% for a single
season-wide average in the original 9-country study); all 11 monthly +
season-max NDVI/EVI columns together carry ~8%. `yield_lag1` still
dominates at 80.3%.

Reproduce with `python src/train.py`.

## Scenario simulation

`src/simulate.py` runs a Monte Carlo scenario per country through the
LightGBM model above: instead of one point forecast, it resamples (5,000
draws, with replacement) that country's own real historical seasons,
stratified by NDVI tercile (`driest_tercile` / `all_years` /
`wettest_tercile`), and reports a yield *distribution* (P10/P50/P90,
probability of falling below last year) rather than a single number.
Scenarios are built by resampling real historical (NDVI, EVI, precip, temp)
tuples together — not by inventing an independent synthetic shock to
precip/temp while leaving NDVI untouched, which would be physically wrong
given how much of the model's signal sits on NDVI/EVI + lag1 (see feature
importance above; a real drought suppresses NDVI too, it's not a separate
independent variable).

Example, Afghanistan:

| Scenario | Pool (real seasons) | P10 | P50 | P90 | P(yield < last year) |
|---|---|---|---|---|---|
| driest tercile | 8 | 1712 | 2005 | 2113 | 100.0% |
| all years | 22 | 1715 | 1981 | 2128 | 91.6% |
| wettest tercile | 8 | 1938 | 2090 | 2179 | 75.4% |

Full output for all 23 countries: `python src/simulate.py` →
`data/processed/simulation_results.csv`.

**Limitation specific to this simulation, disclosed rather than smoothed
over:** for several countries the most recent observed yield is close to
that country's historical high — a real, sustained upward trend
(agricultural intensification) — while the simulator resamples NDVI/weather
from the *full* historical record, not just recent years. That inflates
`P(yield < last year)` for those countries, mixing two different things
together: "a below-trend season is likely" and "the simulator under-weights
the secular trend." Read those numbers as directional, not calibrated
probabilities, until the pool is restricted to a recent window.

## Known limitations

Read this before citing the results anywhere:

- **Fixed Feb-Jun season window for every country**, not a per-country
  wheat-calendar. Real planting/harvest timing varies (winter vs. spring
  wheat, latitude, hemisphere — Argentina and Australia are in the southern
  hemisphere, where Feb-Jun is not really "spring green-up" the way it is
  for the rest of the panel). This is approximate, not phenology-matched.
- **Country-level spatial averaging** blends non-cropland pixels (desert,
  mountains, urban) into the satellite signal — diluted relative to a
  cropland-masked extraction. Worse for continental-scale countries: China,
  USA, Canada, and Australia are reduced at a coarser 5 km scale (vs. 250 m
  for the rest of the panel) purely to keep Earth Engine computation
  tractable — see `src/extract_satellite.py`'s `COARSE_SCALE_COUNTRIES`.
- **Weather is a single point per country** (one grain-belt coordinate), not
  an area average, because NASA POWER is a point API — a bigger
  simplification for Canada or Australia than for the Netherlands.
- **Small N relative to feature count.** 391 training rows, 22 features.
  Bigger than the original 9-country study's ~150, still modest. Simulator
  pools go as low as 8 seasons per country.
- **No farmer-reported ground truth anywhere in this pipeline** — labels are
  national statistics, not field-level outcomes.

## Dashboards

Both served from this repo's own GitHub Pages (`docs/`) — no third-party branding, just the pages:

- **[Harvest From Orbit](https://maiwandalamzoi.github.io/cereal-yield-satellite-forecast/report.html)** — the full results report: pipeline, model comparison, feature importance, Monte Carlo simulator, per-country yield trends.
- **[Grain Belt Atlas](https://maiwandalamzoi.github.io/cereal-yield-satellite-forecast/atlas.html)** — an interactive world map. Click any country to see exactly where its data came from (the real satellite footprint and weather-station coordinates), its yield trend, and its drought-scenario risk. Built with `src/build_map_paths.py` — see that file's docstring for how the map itself was generated.

## Interactive demo

`streamlit_app.py` puts a UI on top of `src/simulate.py` — pick a country
and a scenario, see the projected yield distribution vs. that country's most
recent actual harvest. Deliberately has no independent rainfall/temperature
slider (see the app's own "Why isn't there a slider?" expander) — the
methodology stayed physically honest even where that made for a less flashy
demo control.

```bash
streamlit run streamlit_app.py
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # only needed to re-extract satellite data
```

`requirements.txt` is pinned to exact versions (Python 3.11.9, scikit-learn
1.8.0, etc.) — the versions the committed `models/*.joblib` were actually
trained with. Not a formality: an unpinned install during verification
grabbed a newer scikit-learn and threw an `InconsistentVersionWarning` on
model load. Pinning was then verified by installing into a clean venv from
a fresh clone and confirming the warning is gone.

**You do not need a GEE account or any API key to use the trained models or
the demo.** `models/*.joblib` and `data/processed/panel.csv` are committed —
`streamlit run streamlit_app.py`, `python src/train.py`, and
`python src/simulate.py` all run immediately after `pip install`. A GEE
service account is only needed to re-run `src/extract_satellite.py` and
pull fresh satellite data.

## Reproduce

```bash
python src/extract_yield_wheat.py  # -> data/raw/wheat_yield_faostat.csv (primary target)
python src/extract_yield.py        # -> data/raw/cereal_yield_worldbank.csv (comparison column)
python src/extract_satellite.py    # -> data/raw/satellite_features.csv (resumable, retries transient GEE errors)
python src/extract_weather.py      # -> data/raw/weather_features.csv
python src/build_dataset.py        # -> data/processed/panel.csv
python src/train.py                # -> models/*.joblib, prints metrics
python src/simulate.py             # -> data/processed/simulation_results.csv
```

`extract_satellite.py` is resumable — a killed or interrupted run picks up
from whichever countries already have a full 23 rows on disk rather than
starting over. Real-world runtime note: the 4 continental-scale countries
took several minutes each even at coarse resolution, and Earth Engine's
"Too many concurrent aggregations" error only went away once each
country-year's ~13 satellite reduceRegion calls were split into 6 smaller
requests instead of 1 bundled one — see the script's own comments for the
full story if you hit it again.

## License

MIT
