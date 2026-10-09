"""Simple univariate ARIMA baseline with chronological holdout evaluation."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def prepare_series(frame: pd.DataFrame, date_column: str, value_column: str):
    data = frame[[date_column, value_column]].copy()
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    data = data.groupby("date", as_index=True)["value"].mean()
    data.index = data.index.normalize()
    data = data.groupby(level=0).mean().asfreq("D")
    if len(data) < 45:
        raise ValueError("At least 45 days of daily history are required for this initial ARIMA baseline.")
    if data.notna().sum() < 40:
        raise ValueError("Insufficient valid measurements.")
    if data.isna().mean() > 0.2:
        raise ValueError("More than 20% of daily observations are missing.")
    # Time interpolation only on the historical series.
    return data.interpolate(method="time").ffill().bfill().astype(float)

def forecast(series: pd.Series, horizon: int, order=(2, 1, 2)):
    if series.nunique() < 3:
        raise ValueError("The selected series has insufficient variation for ARIMA.")
    model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit()
    prediction = fit.get_forecast(steps=horizon)
    mean = prediction.predicted_mean
    ci = prediction.conf_int(alpha=0.05)
    result = pd.DataFrame({
        "date": mean.index,
        "predicted": mean.to_numpy(),
        "lower_95": ci.iloc[:, 0].to_numpy(),
        "upper_95": ci.iloc[:, 1].to_numpy(),
    })
    return result, fit.aic

def backtest(series: pd.Series, order=(2, 1, 2)):
    test_size = min(30, max(7, int(len(series) * 0.15)))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    fitted = ARIMA(train, order=order, enforce_stationarity=False,
                   enforce_invertibility=False).fit()
    predicted = fitted.forecast(steps=len(test)).to_numpy()
    actual = test.to_numpy()
    return {
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "Baseline MAE": float(np.mean(np.abs(actual - train.iloc[-1]))),
    }
