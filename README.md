# 🌦️ Weather Prediction AI

Streamlit application for exploring historical weather data and generating **experimental ARIMA time-series forecasts**.

## Features
- Search worldwide cities using Open-Meteo geocoding.
- Retrieve daily historical data with the Open-Meteo Archive API.
- Upload your own CSV or Excel dataset.
- Map date and numeric target columns.
- Train configurable univariate ARIMA and evaluate a chronological holdout.
- View a forecast chart, approximate 95% intervals, and download a CSV.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy
Connect this repository at https://share.streamlit.io/ and choose `app.py` as the main file.

## Uploaded dataset
Your CSV/XLSX should have a date column plus one or more numeric daily measurements; for example:

```csv
date,temperature_c,rainfall_mm
2025-01-01,8.2,0
2025-01-02,7.1,2.3
```

This initial baseline needs at least 45 days of observations and can interpolate limited missing values.

## Limitations
- ARIMA predicts one numeric variable at a time and does not directly predict sunny/cloudy/rainy labels.
- For rainfall, plain ARIMA can produce negative numeric predictions. Do not interpret them as physical rainfall.
- Historical API data may be reanalysis rather than station measurements.
- Seasonal models, better rainfall treatment, rolling backtests, and weather-condition classification are planned enhancements.
- Do not use these statistical forecasts for safety-critical decisions.
