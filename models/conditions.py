"""Experimental daily weather-category prediction using lagged historical codes."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

LABELS = {"Clear": "☀️", "Cloudy": "☁️", "Rainy": "🌧️", "Snowy": "❄️"}
def weather_category(code):
    if pd.isna(code): return None
    code = int(code)
    if code in (0, 1): return "Clear"
    if code in (2, 3, 45, 48): return "Cloudy"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "Rainy"
    if code in (71, 73, 75, 77, 85, 86): return "Snowy"
    return None

def features(day, previous):
    angle = 2 * np.pi * day.dayofyear / 365.25
    return [np.sin(angle), np.cos(angle)] + [int(previous == label) for label in LABELS]

def predict_conditions(frame, date_column="date", code_column="weather_code", days=7):
    if code_column not in frame.columns:
        raise ValueError("A historical WMO weather_code column is required for condition classification.")
    history = frame[[date_column, code_column]].copy()
    history.columns = ["date", "weather_code"]
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["weather_code"] = pd.to_numeric(history["weather_code"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    history["category"] = history["weather_code"].map(weather_category)
    history = history.dropna(subset=["category"]).drop_duplicates("date", keep="last")
    history = history.set_index("date").asfreq("D")
    # Only consecutive observed days are used for training.
    X, y = [], []
    for i in range(1, len(history)):
        prev, current = history["category"].iloc[i - 1], history["category"].iloc[i]
        if pd.notna(prev) and pd.notna(current):
            X.append(features(history.index[i], prev))
            y.append(current)
    if len(X) < 90 or len(set(y)) < 2:
        raise ValueError("At least 90 valid consecutive daily observations and two weather categories are needed.")
    X, y = np.asarray(X), np.asarray(y)
    split = int(len(y) * 0.8)
    if split < 60 or len(set(y[:split])) < 2:
        raise ValueError("Insufficient training data for a chronological holdout.")
    evaluation = RandomForestClassifier(n_estimators=120, max_depth=8,
                                        min_samples_leaf=4, random_state=42, n_jobs=-1)
    evaluation.fit(X[:split], y[:split])
    truth, estimated = y[split:], evaluation.predict(X[split:])
    accuracy = accuracy_score(truth, estimated)
    balanced = balanced_accuracy_score(truth, estimated)
    model = RandomForestClassifier(n_estimators=120, max_depth=8,
                                   min_samples_leaf=4, random_state=42, n_jobs=-1)
    model.fit(X, y)
    previous = history["category"].dropna().iloc[-1]
    last_date = history["category"].dropna().index[-1]
    rows = []
    for offset in range(1, days + 1):
        when = last_date + pd.Timedelta(days=offset)
        probs = model.predict_proba([features(when, previous)])[0]
        pos = int(np.argmax(probs))
        previous = model.classes_[pos]
        rows.append({"date": when, "condition": previous, "icon": LABELS[previous],
                     "model_probability": round(float(probs[pos]), 3)})
    return pd.DataFrame(rows), {"holdout_accuracy": accuracy,
                               "holdout_balanced_accuracy": balanced, "test_days": len(truth),
                               "history_end": last_date}
