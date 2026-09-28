# AI Demand Forecasting & Inventory Optimization

An end-to-end portfolio project connecting **demand forecasting** to a practical **inventory replenishment policy**.

## Business Problem

Supply-chain planners need a forward-looking view of demand to decide how much inventory to hold and when to reorder. Poor forecasts can contribute to stockouts, excess inventory, and unnecessary working capital.

This project builds a workflow that:

1. prepares historical demand data;
2. creates time-series features;
3. compares a seasonal-naive baseline with a machine-learning model;
4. validates forecasts with a time-based split; and
5. converts the forecast into safety-stock, reorder-point, and order-quantity recommendations.

## Dataset

The project supports the **Walmart Store Sales Forecasting** dataset from the Kaggle competition:

https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

The competition data contains weekly sales at store/department level plus additional contextual fields. The original dataset is not redistributed in this repository.

The repository includes a deterministic **synthetic retail-demand dataset** so the project can be run immediately. Synthetic-demo results are clearly separated from any future real-dataset run.

## Forecasting Methodology

### Features

- Lagged demand: 1, 2, 4, 8, 13, and 26 weeks
- Rolling mean and standard deviation
- Week-of-year, month, and year
- Cyclical seasonality features
- Store and product identifiers

### Models

- **Seasonal Naive (13-week)** baseline
- **HistGradientBoostingRegressor** machine-learning model

### Metrics

- MAE
- RMSE
- WAPE

## Inventory Optimization Layer

Forecasts are converted into:

- Lead-time demand
- Safety stock
- Reorder point
- Recommended order quantity
- Inventory risk flag

The included demo uses a **2-week lead time** and **95% service level** as explicit modeling assumptions. These are not observed supplier parameters.

## Project Structure

```text
ai-demand-forecasting-inventory-optimization/
├── data/
│   ├── demo/
│   │   └── demo_demand.csv
│   └── raw/
│       └── README.md
├── notebooks/
├── reports/
│   ├── figures/
│   ├── forecast_validation.csv
│   ├── inventory_recommendations.csv
│   └── metrics.csv
├── src/
│   ├── data_prep.py
│   ├── forecast.py
│   └── inventory.py
├── run_project.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Run the Demo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python run_project.py
```

## Run with the Real Walmart Dataset

1. Download the competition `train.csv`.
2. Put it in `data/raw/train.csv`.
3. Run:

```bash
python run_project.py --real-data
```

## Outputs

- `reports/metrics.csv` — model comparison
- `reports/forecast_validation.csv` — actual vs forecast values
- `reports/inventory_recommendations.csv` — replenishment recommendations
- `reports/figures/forecast_validation.png` — forecast visualization

## Demo Results

The included synthetic demo produces the following holdout results on the final 13 weeks:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal Naive (13-week) | 26.71 | 33.70 | 8.17% |
| HistGradientBoostingRegressor | 22.82 | 27.68 | 6.98%|

On the demo data, the machine-learning model reduces WAPE by approximately **14.5% versus the seasonal baseline**. This is a synthetic demonstration result, not a Walmart performance claim.

## Business Interpretation

The key design principle is **forecast → inventory decision**. Forecasting is treated as a decision-support component rather than an isolated machine-learning exercise.

## Next Steps

- Add walk-forward cross-validation.
- Add promotions and holiday features for the full Walmart dataset.
- Compare Random Forest, XGBoost, and dedicated time-series models.
- Build a Power BI dashboard from the generated CSV outputs.
- Add cost-based order optimization and service-level scenarios.

## Author

**Abdulrhman Yasser Salah**
