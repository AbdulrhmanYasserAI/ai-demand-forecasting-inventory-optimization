from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "data" / "demo" / "demo_demand.csv"
RAW_PATH = ROOT / "data" / "raw" / "train.csv"


def make_demo_data(seed: int = 42) -> pd.DataFrame:
    """Create a deterministic retail-demand dataset for the portfolio demo."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-07", periods=156, freq="W-FRI")
    products = ["P001", "P002", "P003"]
    stores = ["S01", "S02"]
    rows = []

    for store_idx, store in enumerate(stores):
        for product_idx, product in enumerate(products):
            base = 180 + 45 * product_idx + 25 * store_idx
            trend = np.linspace(0, 70 + 10 * product_idx, len(dates))
            seasonal = 28 * np.sin(2 * np.pi * np.arange(len(dates)) / 13)
            holiday = np.array(
                [24 if d.month in (11, 12) and d.day <= 21 else 0 for d in dates]
            )
            noise = rng.normal(0, 15 + product_idx * 3, len(dates))
            demand = np.maximum(25, base + trend + seasonal + holiday + noise)

            for date, value in zip(dates, demand):
                rows.append(
                    {
                        "Store": store,
                        "Product": product,
                        "Date": date,
                        "Demand": round(float(value), 2),
                    }
                )

    return pd.DataFrame(rows)


def ensure_demo_file() -> pd.DataFrame:
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEMO_PATH.exists():
        df = make_demo_data()
        df.to_csv(DEMO_PATH, index=False)
    else:
        df = pd.read_csv(DEMO_PATH, parse_dates=["Date"])
    return df


def load_data(use_real_data: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the optional real dataset or the included demo dataset."""
    if use_real_data and RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH, parse_dates=["Date"])
        df = df.rename(columns={"Weekly_Sales": "Demand"})
        df["Store"] = df["Store"].astype(str)
        df["Product"] = df["Dept"].astype(str).map(lambda x: f"D{x}")
        required = {"Store", "Product", "Date", "Demand"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Real dataset is missing columns: {sorted(missing)}")
        clean = (
            df[["Store", "Product", "Date", "Demand"]]
            .dropna()
            .sort_values(["Store", "Product", "Date"])
            .reset_index(drop=True)
        )
        return clean, "walmart"

    return ensure_demo_file(), "demo"
