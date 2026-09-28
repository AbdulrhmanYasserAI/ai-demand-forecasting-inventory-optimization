from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd


def inventory_policy(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    lead_time_weeks: int = 2,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """Convert demand forecasts into a simple transparent replenishment policy."""
    z = NormalDist().inv_cdf(service_level)

    history = history.sort_values(["Store", "Product", "Date"]).copy()
    demand_std = history.groupby(["Store", "Product"])["Demand"].std().rename(
        "demand_std"
    )

    recent_mean = (
        history.groupby(["Store", "Product"])
        .tail(13)
        .groupby(["Store", "Product"])["Demand"]
        .mean()
        .rename("recent_avg_demand")
    )

    out = forecast.copy()
    out = out.merge(demand_std, on=["Store", "Product"], how="left")
    out = out.merge(recent_mean, on=["Store", "Product"], how="left")
    out["demand_std"] = out["demand_std"].fillna(out["recent_avg_demand"] * 0.15)

    out["lead_time_demand"] = out["Forecast_Demand"] * lead_time_weeks
    out["safety_stock"] = (
        z * out["demand_std"] * np.sqrt(lead_time_weeks)
    )
    out["reorder_point"] = out["lead_time_demand"] + out["safety_stock"]
    out["recommended_order_qty"] = np.maximum(
        0, out["reorder_point"] - out["on_hand_inventory"]
    )
    out["risk_flag"] = np.select(
        [
            out["on_hand_inventory"] < out["reorder_point"],
            out["on_hand_inventory"] < out["Forecast_Demand"],
        ],
        ["REORDER", "WATCH"],
        default="HEALTHY",
    )

    return out
