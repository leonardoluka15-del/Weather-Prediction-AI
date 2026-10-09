"""Open-Meteo geocoding and historical daily weather data."""
from datetime import date
import pandas as pd
import requests

GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
             "precipitation_sum", "wind_speed_10m_max", "weather_code"]

def find_locations(query: str):
    response = requests.get(GEOCODING, params={"name": query, "count": 10, "language": "en", "format": "json"}, timeout=20)
    response.raise_for_status()
    return response.json().get("results", [])

def get_historical(latitude: float, longitude: float, start: date, end: date):
    if start >= end:
        raise ValueError("Start date must precede end date.")
    response = requests.get(ARCHIVE, params={
        "latitude": latitude, "longitude": longitude,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(VARIABLES), "timezone": "auto",
    }, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload:
        raise ValueError(payload.get("reason", "No historical daily observations returned."))
    frame = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame
