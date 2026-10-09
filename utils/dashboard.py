"""Combine separately generated forecast results by exact forecast date."""
import pandas as pd

DISPLAY = {
    "temperature_2m_mean": ("Mean temperature", "°C"),
    "temperature_2m_max": ("High", "°C"),
    "temperature_2m_min": ("Low", "°C"),
    "precipitation_sum": ("Rainfall", "mm"),
    "wind_speed_10m_max": ("Wind", "km/h"),
    "relative_humidity_2m_mean": ("Humidity", "%"),
}

def combine_forecasts(numeric=None, conditions=None):
    """Outer-merge by date; never align results by row index or invent missing values."""
    if numeric is None and conditions is None:
        return pd.DataFrame()
    output = None
    if numeric is not None and not numeric.empty:
        numeric = numeric.copy()
        numeric["date"] = pd.to_datetime(numeric["date"]).dt.normalize()
        output = numeric.pivot_table(index="date", columns="variable", values="predicted",
                                     aggfunc="first").reset_index()
    if conditions is not None and not conditions.empty:
        cond = conditions[["date", "condition", "icon", "model_probability"]].copy()
        cond["date"] = pd.to_datetime(cond["date"]).dt.normalize()
        output = cond if output is None else output.merge(cond, on="date", how="outer")
    if output is None:
        return pd.DataFrame()
    return output.sort_values("date").reset_index(drop=True)

def display_value(row, name):
    value = row.get(name)
    if pd.isna(value):
        return "—"
    _, unit = DISPLAY[name]
    return f"{value:.1f} {unit}"
