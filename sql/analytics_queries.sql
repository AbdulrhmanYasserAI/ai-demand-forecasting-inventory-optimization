-- Example SQL analytics layer for the generated project outputs.
-- These queries assume the CSV outputs have been loaded into a SQL table/view.

-- 1) Highest forecast demand by store/product
SELECT
    Store,
    Product,
    AVG(Forecast_Demand) AS Avg_Forecast_Demand
FROM forecast_validation
GROUP BY Store, Product
ORDER BY Avg_Forecast_Demand DESC;

-- 2) Forecast accuracy by product
SELECT
    Product,
    AVG(Absolute_Error) AS MAE_Proxy
FROM forecast_validation
GROUP BY Product
ORDER BY MAE_Proxy DESC;

-- 3) Replenishment actions
SELECT
    Store,
    Product,
    Forecast_Demand,
    reorder_point,
    recommended_order_qty,
    risk_flag
FROM inventory_recommendations
WHERE risk_flag = 'REORDER'
ORDER BY recommended_order_qty DESC;
