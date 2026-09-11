"""
Interactive demo on top of src/simulate.py's Monte Carlo scenario simulator.

Deliberately does NOT expose independent precip/temp/NDVI sliders. The
trained model puts ~95% of its signal on NDVI + last-year's yield and under
2% combined on precip/temp (see README feature-importance table) — letting
someone drag a precip slider while NDVI stays frozen at its historical
average would produce a chart that barely moves, which looks like a bug but
is actually the honest behavior of this specific model. So the one control
that changes the underlying data is a scenario tercile (driest / average /
wettest historical seasons for that country), which resamples real
historical (NDVI, EVI, precip, temp) tuples together — same method as
simulate.py, reused directly rather than reimplemented.

Run: streamlit run streamlit_app.py
"""
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from simulate import FEATURES, SCENARIOS, simulate_country  # noqa: E402

st.set_page_config(page_title="Wheat Yield Scenario Simulator", page_icon="🌾")

st.title("🌾 Wheat Yield Scenario Simulator")
st.caption(
    "Monte Carlo scenarios built on a trained Random Forest model — "
    "[see the full methodology, metrics, and honest limitations on GitHub]"
    "(https://github.com/maiwandalamzoi/cereal-yield-satellite-forecast)."
)


@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_yield.joblib")


@st.cache_data
def load_panel():
    return pd.read_csv("data/processed/panel.csv")


model = load_model()
panel = load_panel()

countries = sorted(panel["country_iso3"].unique())
country_names = panel.drop_duplicates("country_iso3").set_index("country_iso3")["country_name"].to_dict()

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", countries, format_func=lambda c: f"{country_names[c]} ({c})")
with col2:
    scenario = st.select_slider(
        "Scenario (real historical seasons for this country, by NDVI tercile)",
        options=SCENARIOS,
        value="all_years",
        format_func=lambda s: {"driest_tercile": "Driest 1/3 of seasons",
                                "all_years": "All seasons (typical)",
                                "wettest_tercile": "Wettest 1/3 of seasons"}[s],
    )

latest_row = panel[panel["country_iso3"] == country].sort_values("year").iloc[-1]
latest_yield = latest_row["wheat_yield_kg_ha"]
latest_year = int(latest_row["year"])

rng = np.random.default_rng(42)
result = simulate_country(model, panel, country, scenario, latest_yield, rng)

if result is None:
    st.error("Not enough historical seasons for this scenario.")
else:
    st.metric(
        f"Projected yield — median (P50)",
        f"{result['p50_kg_ha']:,.0f} kg/ha",
        delta=f"{result['p50_kg_ha'] - latest_yield:+,.0f} kg/ha vs. {latest_year} actual",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("P10 (pessimistic)", f"{result['p10_kg_ha']:,.0f} kg/ha")
    c2.metric("P50 (median)", f"{result['p50_kg_ha']:,.0f} kg/ha")
    c3.metric("P90 (optimistic)", f"{result['p90_kg_ha']:,.0f} kg/ha")

    st.caption(
        f"P(yield < {latest_year} actual of {latest_yield:,.0f} kg/ha) = "
        f"**{result['prob_below_last_year']:.0%}** — based on resampling "
        f"{result['n_historical_seasons_in_pool']} real historical seasons "
        f"(2002-2023), 5,000 draws."
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    pool = panel[panel["country_iso3"] == country]
    if scenario != "all_years":
        terciles = pool["ndvi_season_mean"].quantile([1 / 3, 2 / 3]).values
        pool = pool[pool["ndvi_season_mean"] <= terciles[0]] if scenario == "driest_tercile" \
            else pool[pool["ndvi_season_mean"] >= terciles[1]]
    x_cols = [c for c in FEATURES if c != "yield_lag1"]
    X = pool[x_cols].copy()
    X["yield_lag1"] = latest_yield
    preds = model.predict(X[FEATURES])
    ax.hist(np.repeat(preds, 200), bins=30, color="#3a7d44", alpha=0.75)
    ax.axvline(latest_yield, color="#b3261e", linestyle="--", label=f"{latest_year} actual")
    ax.set_xlabel("kg/ha")
    ax.set_yticks([])
    ax.legend()
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)

    if scenario != "all_years":
        st.info(
            f"This scenario resamples only **{result['n_historical_seasons_in_pool']} real "
            f"seasons** from {country}'s 2002-2023 record — a small-sample bootstrap, not a "
            f"parametric extreme-event model. It can't represent a scenario worse/better than "
            f"anything actually observed in that record. Full disclosure in the README."
        )

with st.expander("Why isn't there a rainfall/temperature slider?"):
    st.write(
        "There was one in an earlier draft — it was removed because it was misleading. "
        "This model's feature importance puts ~95% of its signal on season-average NDVI "
        "and last year's yield, under 2% combined on precipitation and temperature at "
        "this level of aggregation (a single grain-belt coordinate per country, not an "
        "area average — see README). An independent rain slider would barely move the "
        "chart, which would look like a broken control rather than what it actually is: "
        "an honest reflection of a coarse, country-level weather feature. The scenario "
        "control above instead resamples real historical seasons where NDVI, EVI, "
        "precipitation, and temperature moved together the way they actually did."
    )
