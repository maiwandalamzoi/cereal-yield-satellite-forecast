# Cereal Yield Forecasting from Satellite + Weather Data

Predicts a country's annual cereal yield from in-season satellite vegetation
indices and weather, across nine wheat-belt countries (Afghanistan, Pakistan,
Iran, Kazakhstan, Uzbekistan, Tajikistan, Turkiye, Ukraine) plus one
out-of-regime comparator (Netherlands).

This is a companion project to
[crop-stress-prediction](https://github.com/maiwandalamzoi/crop-stress-prediction):
that project detects *anomalies* in vegetation health 2-4 weeks ahead; this
one asks a different, harder question — can in-season satellite + weather
signal actually forecast the *yield number* at harvest?

**What this is:** a real, trained regression model (XGBoost / Random Forest)
evaluated on a genuine train/test split, benchmarked against a naive
lag-1 baseline, with honestly reported metrics.
**What this is not:** a production yield forecaster. See *Known
limitations* below before drawing conclusions from the numbers.

## Method

1. **Labels** — annual cereal yield (kg/ha) per country, from the World Bank
   Open Data API (`AG.YLD.CREL.KG`), 2001-2023. `src/extract_yield.py`
2. **Satellite features** — country-level seasonal NDVI/EVI from MODIS
   MOD13Q1 (250 m, 16-day composite), averaged over a fixed Feb-Jun growing
   season window, reduced over each country's boundary via Google Earth
   Engine. `src/extract_satellite.py`
3. **Weather features** — season-total precipitation and mean temperature
   from NASA POWER, at one representative grain-belt coordinate per country
   (not a country-wide average). `src/extract_weather.py`
4. **Panel + lag feature** — merged into one country×year panel with a
   `yield_lag1` feature (that country's own yield the prior year — known at
   prediction time, not a leak). `src/build_dataset.py`
5. **Train/evaluate** — time-based split (train ≤2018, test 2019-2023, no
   random k-fold — a random split would leak future climate patterns into
   training). Six models compared: naive lag1 baseline, Ridge, a country
   fixed-effects linear model, Random Forest, XGBoost, LightGBM.
   `src/train.py`
6. **Scenario simulation** — Monte Carlo drought/heatwave scenarios: sample
   weather from each country's own historical distribution (optionally
   shifted to represent a scenario), run the trained model thousands of
   times, report a yield *distribution* (P10/P50/P90) instead of one number.
   `src/simulate.py`

## Results

Train: 153 rows (2002–2018). Test: 45 rows (2019–2023), a genuine holdout —
none of these years' weather/satellite/yield were seen during training.

| Model | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---|---|---|---|
| baseline (yield = last year) | 299.5 | 421.7 | 0.954 |
| ridge | 306.6 | 400.2 | 0.959 |
| **fixed-effects (country dummies)** | **276.1** | **368.0** | **0.965** |
| random forest | 266.2 | 370.2 | 0.965 |
| xgboost | 295.4 | 409.8 | 0.957 |
| lightgbm | 461.8 | 735.6 | 0.861 |

**Honest read of this table:** the naive "this year repeats last year"
baseline is already strong (R²=0.954) — most of a country's cereal yield
*is* just its recent level, which is exactly why `yield_lag1` dominates
feature importance below. The best model (country fixed-effects, tied with
random forest) beats it by ~13% RMSE, not by a wide margin. That's a real
but modest lift from adding in-season satellite/weather signal on top of
persistence — closer to crop-stress-prediction's honest F1 0.405 than to a
headline number. LightGBM's default hyperparameters (`min_child_samples=20`,
`num_leaves=31`) are tuned for much larger datasets than this one's 153
training rows and it shows — no tuning was done to make this look better,
it's reported as-is.

XGBoost feature importance:

| Feature | Importance |
|---|---|
| `ndvi_season_mean` | 0.547 |
| `yield_lag1` | 0.424 |
| `evi_season_mean` | 0.010 |
| `season_temp_mean_c` | 0.008 |
| `season_precip_total_mm` | 0.006 |
| `ndvi_season_max` | 0.005 |

Season-mean NDVI and last year's yield carry nearly all the signal; the
country-average weather point contributes almost nothing at this level of
aggregation — consistent with the "single point ≠ country average" limitation
above.

Reproduce with `python src/train.py`.

## Scenario simulation

`src/simulate.py` runs a Monte Carlo scenario per country: instead of one
point forecast, it resamples (5,000 draws, with replacement) that country's
own real historical seasons, stratified by NDVI tercile (`driest_tercile` /
`all_years` / `wettest_tercile`), and reports a yield *distribution*
(P10/P50/P90, probability of falling below last year) rather than a single
number. Scenarios are built by resampling real historical (NDVI, EVI,
precip, temp) tuples together — not by inventing an independent synthetic
shock to precip/temp while leaving NDVI untouched, which would be physically
wrong given the model puts ~95% of its signal on NDVI + lag1 and under 2%
combined on precip/temp (see feature importance above; a real drought
suppresses NDVI too, it's not a separate independent variable).

Example, Afghanistan:

| Scenario | Pool (real seasons) | P10 | P50 | P90 | P(yield < last year) |
|---|---|---|---|---|---|
| driest tercile | 8 | 2117 | 2308 | 2318 | 49.7% |
| all years | 22 | 2173 | 2315 | 2368 | 27.2% |
| wettest tercile | 8 | 2333 | 2366 | 2467 | 0.0% |

Full output for all 9 countries: `python src/simulate.py` →
`data/processed/simulation_results.csv`.

**Limitation specific to this simulation, disclosed rather than smoothed
over:** for several countries (Tajikistan, Turkiye, Ukraine, Uzbekistan) the
most recent observed yield is close to that country's historical high — a
real, sustained upward trend (agricultural intensification) — while the
simulator resamples NDVI/weather from the *full* historical record, not just
recent years. That produces `P(yield < last year) ≈ 100%` for those
countries, which mixes two different things together: "a below-trend season
is likely" and "the simulator under-weights the secular trend." Read those
numbers as directional, not calibrated probabilities, until the pool is
restricted to a recent window.

## Known limitations

Read this before citing the results anywhere:

- **Cereal yield, not wheat-specific.** World Bank's `AG.YLD.CREL.KG` covers
  all cereals. FAOSTAT has a wheat-only series (item 15, element 5419) but
  `fenixservices.fao.org` was returning HTTP 521 (server down) while this
  was built — swapping in FAOSTAT once it's back is the natural refinement.
  Afghanistan's cereal area is >80% wheat, so this is a defensible proxy,
  not an exact match, for that country; less so for some others in the
  panel.
- **Fixed Feb-Jun season window for every country**, not a per-country
  wheat-calendar. Real planting/harvest timing varies (winter vs. spring
  wheat, latitude). This is approximate, not phenology-matched.
- **Country-level spatial averaging** blends non-cropland pixels (desert,
  mountains, urban) into the satellite signal — it is diluted relative to a
  cropland-masked extraction.
- **Weather is a single point per country** (one grain-belt coordinate), not
  an area average, because NASA POWER is a point API.
- **Small N.** 9 countries × ~22 years ≈ 200 rows before the lag1 dropna,
  ~150-170 after train/test split. Enough to compare models honestly, not
  enough for strong generalization claims.
- **No farmer-reported ground truth anywhere in this pipeline** — labels are
  national statistics, not field-level outcomes.

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
python src/extract_yield.py        # -> data/raw/cereal_yield_worldbank.csv
python src/extract_satellite.py    # -> data/raw/satellite_features.csv
python src/extract_weather.py      # -> data/raw/weather_features.csv
python src/build_dataset.py        # -> data/processed/panel.csv
python src/train.py                # -> models/*.joblib, prints metrics
python src/simulate.py             # -> data/processed/simulation_results.csv
```

## License

MIT
