"""
Train + evaluate cereal-yield regressors on data/processed/panel.csv.

Time-based split (not random k-fold): train on years <= TRAIN_END_YEAR,
test on the years after it. This mirrors crop-stress-prediction's
methodology -- a random split would leak future years' climate patterns
into training and overstate accuracy.

Models compared, in increasing complexity:
  1. baseline_lag1   -- naive: predict this year = last year (no model at all)
  2. ridge            -- regularized linear regression on the 6 physical features
  3. fixed_effects     -- linear regression + one dummy per country (lets each
                          country have its own intercept instead of asking the
                          physical features to explain baseline level differences)
  4. random_forest
  5. xgboost
  6. lightgbm

With a few hundred rows, simpler models are not a formality here -- they
are a real contender, not just a strawman baseline. "Best model" is
decided by test-set RMSE/R2 below, not by which one sounds most
sophisticated.

Features use the monthly Feb-Jun NDVI/EVI/precip/temp breakdown, NOT also
the season-mean columns -- ndvi_season_mean and evi_season_mean are
arithmetic means of their own 5 monthly columns, so including both would
be exact linear redundancy (rank-deficient for the fixed-effects OLS
model specifically). ndvi_season_max is kept: max isn't a linear function
of the monthly means, so it carries information the monthly columns don't.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm

TRAIN_END_YEAR = 2018  # train 2002-2018, test 2019-2023

MONTHS = [2, 3, 4, 5, 6]
FEATURES = (
    [f"ndvi_m{m:02d}" for m in MONTHS]
    + [f"evi_m{m:02d}" for m in MONTHS]
    + [f"precip_m{m:02d}_mm" for m in MONTHS]
    + [f"temp_m{m:02d}_c" for m in MONTHS]
    + ["ndvi_season_max", "yield_lag1"]
)
TARGET = "wheat_yield_kg_ha"


def evaluate(name, y_true, y_pred, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:22s}  MAE={mae:8.1f} kg/ha  RMSE={rmse:8.1f} kg/ha  R2={r2:6.3f}")
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = pd.read_csv("data/processed/panel.csv").dropna(subset=FEATURES + [TARGET])
    train = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    test = df[df["year"] > TRAIN_END_YEAR].reset_index(drop=True)
    print(f"train: {len(train)} rows (years {train['year'].min()}-{train['year'].max()})")
    print(f"test:  {len(test)} rows (years {test['year'].min()}-{test['year'].max()})")
    print()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = {}
    models = {}

    # 1. Naive baseline: this year = last year
    evaluate("baseline_lag1", y_test, test["yield_lag1"], results)

    # 2. Ridge on standardized monthly physical features (no country identity)
    scaler = StandardScaler().fit(X_train)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)
    evaluate("ridge", y_test, ridge.predict(scaler.transform(X_test)), results)
    models["ridge"] = (ridge, scaler)

    # 3. Country fixed-effects linear model: physical features + one dummy
    #    per country, so each country gets its own intercept instead of the
    #    model inferring "Netherlands is just structurally high-yield" from
    #    6 numbers.
    fe_train = pd.get_dummies(train[["country_iso3"] + FEATURES], columns=["country_iso3"], drop_first=True)
    fe_test = pd.get_dummies(test[["country_iso3"] + FEATURES], columns=["country_iso3"], drop_first=True)
    fe_test = fe_test.reindex(columns=fe_train.columns, fill_value=0)
    fe_model = sm.OLS(y_train.values, sm.add_constant(fe_train.astype(float))).fit()
    fe_pred = fe_model.predict(sm.add_constant(fe_test.astype(float), has_constant="add"))
    evaluate("fixed_effects", y_test, fe_pred, results)
    models["fixed_effects"] = (fe_model, list(fe_train.columns))

    # 4. Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("random_forest", y_test, rf.predict(X_test), results)
    models["random_forest"] = rf

    # 5. XGBoost
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        random_state=42, reg_lambda=1.0)
    xgb.fit(X_train, y_train)
    evaluate("xgboost", y_test, xgb.predict(X_test), results)
    models["xgboost"] = xgb

    # 6. LightGBM
    lgbm = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    evaluate("lightgbm", y_test, lgbm.predict(X_test), results)
    models["lightgbm"] = lgbm

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nbest model by test RMSE: {best_name}  "
          f"(RMSE={results[best_name]['rmse']:.1f}, R2={results[best_name]['r2']:.3f})")

    print("\nfeature importance (xgboost):")
    for feat, imp in sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:24s} {imp:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgboost_yield.joblib")
    joblib.dump(rf, "models/random_forest_yield.joblib")
    joblib.dump(lgbm, "models/lightgbm_yield.joblib")
    joblib.dump({"model": ridge, "scaler": scaler, "features": FEATURES}, "models/ridge_yield.joblib")
    print("\nsaved models/*.joblib")

    return results, best_name


if __name__ == "__main__":
    main()
