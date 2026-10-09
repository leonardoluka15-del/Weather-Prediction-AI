"""Independent multi-target statistical forecasts (not a joint multivariate model)."""
import pandas as pd
from models.forecasting import prepare_series, compare_models, backtest, forecast

TARGETS = {
    "temperature_2m_mean": "Temperature (°C)",
    "temperature_2m_max": "Maximum temperature (°C)",
    "temperature_2m_min": "Minimum temperature (°C)",
    "precipitation_sum": "Precipitation (mm)",
    "wind_speed_10m_max": "Maximum wind (km/h)",
    "relative_humidity_2m_mean": "Humidity (%)",
}

def forecast_many(frame, date_column, target_columns, horizon, order, choice):
    tables, metrics, failures = [], [], []
    for column in target_columns:
        try:
            series = prepare_series(frame, date_column, column)
            comparison = None
            if choice.startswith("Automatic"):
                winner, comparison, _ = compare_models(series, order)
                model = winner["model"]
                seasonal = winner["seasonal_order"]
                score = {k: winner[k] for k in ("MAE", "RMSE", "Baseline MAE")}
            else:
                model = choice
                seasonal = (1, 0, 0, 7) if choice.startswith("SARIMA") else (0, 0, 0, 0)
                score = backtest(series, order, seasonal)
            predicted, aic = forecast(series, horizon, order, seasonal)
            predicted["variable"] = column
            predicted["model"] = model
            # A physical lower bound is enforced only on displayed point estimates,
            # not on statistical confidence intervals (which remain model outputs).
            if "precip" in column.lower() or "rain" in column.lower() or "wind" in column.lower():
                predicted["predicted"] = predicted["predicted"].clip(lower=0)
            if "humid" in column.lower():
                predicted["predicted"] = predicted["predicted"].clip(lower=0, upper=100)
            tables.append(predicted)
            metrics.append({"Variable": column, "Model": model, "MAE": score["MAE"],
                            "RMSE": score["RMSE"], "Naive MAE": score["Baseline MAE"],
                            "AIC": aic})
        except Exception as exc:
            failures.append(f"{column}: {exc}")
    if not tables:
        raise ValueError("No variable could be forecast. " + "; ".join(failures))
    return pd.concat(tables, ignore_index=True), pd.DataFrame(metrics), failures
