from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LAGS = [1, 2, 4, 8, 13, 26]
ROLLS = [4, 8, 13]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["Store", "Product", "Date"])
    group = out.groupby(["Store", "Product"], sort=False)

    for lag in LAGS:
        out[f"lag_{lag}"] = group["Demand"].shift(lag)

    for window in ROLLS:
        shifted = group["Demand"].shift(1)
        out[f"roll_mean_{window}"] = (
            shifted.groupby([out["Store"], out["Product"]], sort=False)
            .transform(lambda s: s.rolling(window).mean())
        )
        out[f"roll_std_{window}"] = (
            shifted.groupby([out["Store"], out["Product"]], sort=False)
            .transform(lambda s: s.rolling(window).std())
        )

    iso_week = out["Date"].dt.isocalendar().week.astype(int)
    out["weekofyear"] = iso_week
    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year
    out["sin_year"] = np.sin(2 * np.pi * out["weekofyear"] / 52.0)
    out["cos_year"] = np.cos(2 * np.pi * out["weekofyear"] / 52.0)
    out["store_code"] = pd.factorize(out["Store"])[0]
    out["product_code"] = pd.factorize(out["Product"])[0]

    return out.dropna().reset_index(drop=True)


def time_split(df: pd.DataFrame, valid_weeks: int = 13):
    cutoff = df["Date"].max() - pd.Timedelta(weeks=valid_weeks - 1)
    train = df[df["Date"] < cutoff].copy()
    valid = df[df["Date"] >= cutoff].copy()
    return train, valid


def seasonal_naive(
    train_raw: pd.DataFrame, valid: pd.DataFrame, season_lag: int = 13
) -> np.ndarray:
    lookup = train_raw.set_index(["Store", "Product", "Date"])["Demand"]
    predictions = []

    for row in valid.itertuples(index=False):
        key = (row.Store, row.Product, row.Date - pd.Timedelta(weeks=season_lag))
        predictions.append(lookup.get(key, np.nan))

    pred = pd.Series(predictions, dtype="float64")
    fallback = float(train_raw["Demand"].mean())
    return pred.fillna(fallback).to_numpy()


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE": float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))),
    }


def train_model(train: pd.DataFrame, valid: pd.DataFrame):
    feature_cols = [
        column
        for column in train.columns
        if column.startswith(("lag_", "roll_"))
        or column
        in {
            "weekofyear",
            "month",
            "year",
            "sin_year",
            "cos_year",
            "store_code",
            "product_code",
        }
    ]

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[feature_cols], train["Demand"])
    predictions = model.predict(valid[feature_cols])
    return model, feature_cols, predictions
