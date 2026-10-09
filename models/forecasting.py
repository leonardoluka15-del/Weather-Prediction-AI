"""Daily univariate ARIMA and weekly-seasonal SARIMA comparison."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prepare_series(frame: pd.DataFrame, date_column: str, value_column: str):
    data = frame[[date_column, value_column]].copy()
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    data = data.groupby("date")["value"].mean()
    data.index = data.index.normalize()
    data = data.groupby(level=0).mean().asfreq("D")
    if len(data) < 45 or data.notna().sum() < 40:
        raise ValueError("At least 45 days of history and 40 valid daily values are required.")
    if data.isna().mean() > 0.2:
        raise ValueError("More than 20% of the daily values are missing.")
    return data.interpolate(method="time").ffill().bfill().astype(float)

def _fit(series, order, seasonal_order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False, maxiter=70)
    if not np.isfinite(fitted.aic):
        raise ValueError("Model fitting yielded a non-finite AIC.")
    return fitted

def forecast(series: pd.Series, horizon: int, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)):
    if series.nunique() < 3:
        raise ValueError("The selected series has insufficient variation.")
    fit = _fit(series, order, seasonal_order)
    prediction = fit.get_forecast(steps=horizon)
    mean = prediction.predicted_mean
    ci = prediction.conf_int(alpha=0.05)
    result = pd.DataFrame({
        "date": mean.index, "predicted": mean.to_numpy(),
        "lower_95": ci.iloc[:, 0].to_numpy(),
        "upper_95": ci.iloc[:, 1].to_numpy(),
    })
    return result, float(fit.aic)

def backtest(series: pd.Series, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)):
    test_size = min(30, max(7, int(len(series) * 0.15)))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    fitted = _fit(train, order, seasonal_order)
    predicted = fitted.forecast(steps=len(test)).to_numpy()
    actual = test.to_numpy()
    return {
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "Baseline MAE": float(np.mean(np.abs(actual - train.iloc[-1]))),
    }

def compare_models(series: pd.Series, order=(2, 1, 2)):
    """Choose the lowest holdout MAE among ARIMA and weekly SARIMA."""
    candidates = [
        ("ARIMA", (0, 0, 0, 0)),
        ("SARIMA (weekly)", (1, 0, 0, 7)),
    ]
    scores, errors = [], []
    for name, seasonal in candidates:
        try:
            metrics = backtest(series, order, seasonal)
            if not np.isfinite(metrics["MAE"]):
                raise ValueError("Non-finite model error")
            scores.append({"model": name, "seasonal_order": seasonal, **metrics})
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not scores:
        raise ValueError("No candidate models fitted. " + "; ".join(errors))
    winner = min(scores, key=lambda x: x["MAE"])
    return winner, scores, errors
