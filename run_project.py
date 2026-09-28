from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import load_data
from src.forecast import add_features, time_split, seasonal_naive, evaluate, train_model
from src.inventory import inventory_policy

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def main(use_real_data: bool = False) -> None:
    df, data_source = load_data(use_real_data=use_real_data)
    featured = add_features(df)
    train, valid = time_split(featured, valid_weeks=13)

    raw_train = df[df["Date"] < valid["Date"].min()].copy()
    naive_pred = seasonal_naive(raw_train, valid)
    naive_metrics = evaluate(valid["Demand"].to_numpy(), naive_pred)

    _, feature_cols, ml_pred = train_model(train, valid)
    ml_metrics = evaluate(valid["Demand"].to_numpy(), ml_pred)

    metrics = pd.DataFrame(
        [
            {"model": "Seasonal Naive (13-week)", **naive_metrics},
            {"model": "HistGradientBoostingRegressor", **ml_metrics},
        ]
    )
    metrics.to_csv(REPORTS / "metrics.csv", index=False)

    validation = valid[["Store", "Product", "Date", "Demand"]].copy()
    validation["Forecast_Demand"] = ml_pred
    validation["Absolute_Error"] = (
        validation["Demand"] - validation["Forecast_Demand"]
    ).abs()
    validation.to_csv(REPORTS / "forecast_validation.csv", index=False)

    last_date = validation["Date"].max()
    latest = validation[validation["Date"] == last_date].copy()
    latest["on_hand_inventory"] = (latest["Forecast_Demand"] * 1.4).round(0)
    recommendations = inventory_policy(
        df[df["Date"] < last_date],
        latest[["Store", "Product", "Date", "Forecast_Demand", "on_hand_inventory"]],
    )
    recommendations.to_csv(REPORTS / "inventory_recommendations.csv", index=False)

    top_key = (
        df.groupby(["Store", "Product"])["Demand"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    raw_series = df[
        (df["Store"] == top_key[0]) & (df["Product"] == top_key[1])
    ]
    valid_series = validation[
        (validation["Store"] == top_key[0])
        & (validation["Product"] == top_key[1])
    ]

    plt.figure(figsize=(11, 5))
    plt.plot(raw_series["Date"], raw_series["Demand"], label="Actual")
    plt.plot(valid_series["Date"], valid_series["Forecast_Demand"], label="Forecast")
    plt.axvline(
        valid_series["Date"].min(),
        linestyle="--",
        label="Validation start",
    )
    plt.title(f"Demand Forecast — {top_key[0]} / {top_key[1]}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "forecast_validation.png", dpi=160)
    plt.close()

    print(f"Data source: {data_source}")
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nForecasting features: {len(feature_cols)}")
    print("\nGenerated outputs:")
    print("- reports/metrics.csv")
    print("- reports/forecast_validation.csv")
    print("- reports/inventory_recommendations.csv")
    print("- reports/figures/forecast_validation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use data/raw/train.csv if the real Walmart dataset is available.",
    )
    args = parser.parse_args()
    main(use_real_data=args.real_data)
