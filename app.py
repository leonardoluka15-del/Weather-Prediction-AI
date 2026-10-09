"""Streamlit entry point for La Météo."""
from datetime import date, timedelta
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.forecasting import prepare_series, forecast, backtest, compare_models
from models.multi_forecast import forecast_many, TARGETS
from models.conditions import predict_conditions
from services.weather_api import find_locations, get_historical

st.set_page_config(page_title="La Météo", page_icon="🌦️", layout="wide")
st.title("🌦️ La Météo")
st.caption("AI-powered weather forecasting · Worldwide locations · Upload or online data")
st.warning("Experimental statistical forecast, not an official weather forecast. Not for safety-critical decisions.")

@st.cache_data(ttl=86400, show_spinner=False)
def cached_locations(query):
    return find_locations(query)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_history(lat, lon, start, end):
    return get_historical(lat, lon, start, end)

source = st.sidebar.radio("Historical data source", ["Online weather data", "Upload CSV / Excel"])
days = st.sidebar.selectbox("Forecast days", [1, 3, 7, 14, 30], index=2)
order = (
    st.sidebar.selectbox("AR order (p)", [0, 1, 2, 3, 4], index=2),
    st.sidebar.selectbox("Differencing (d)", [0, 1, 2], index=1),
    st.sidebar.selectbox("MA order (q)", [0, 1, 2, 3, 4], index=2),
)
model_choice = st.sidebar.radio("Forecast model", ["Automatic (compare ARIMA and SARIMA)", "ARIMA", "SARIMA (weekly)"])
frame = None
if source == "Online weather data":
    query = st.text_input("Search a city worldwide", value="Nantes")
    if len(query.strip()) >= 2:
        try:
            locations = cached_locations(query.strip())
            if not locations:
                st.warning("No matching locations. Try a larger city or another spelling.")
            else:
                labels = [
                    f"{x['name']}, {x.get('admin1', '')}, {x.get('country', '')} "
                    f"({x['latitude']:.3f}, {x['longitude']:.3f})"
                    for x in locations
                ]
                selected = locations[st.selectbox("Select location", range(len(labels)),
                                                  format_func=lambda i: labels[i])]
                start = st.date_input("History starts", value=date.today() - timedelta(days=365 * 5),
                                      max_value=date.today() - timedelta(days=7))
                end = st.date_input("History ends", value=date.today() - timedelta(days=7),
                                    max_value=date.today() - timedelta(days=2))
                if st.button("Fetch historical data", type="primary"):
                    with st.spinner("Retrieving historical daily data..."):
                        st.session_state["weather_history"] = cached_history(
                            selected["latitude"], selected["longitude"], start, end)
                        st.session_state["weather_source"] = labels[labels.index(
                            f"{selected['name']}, {selected.get('admin1', '')}, {selected.get('country', '')} "
                            f"({selected['latitude']:.3f}, {selected['longitude']:.3f})")]
                if st.session_state.get("weather_source") == labels[labels.index(
                    f"{selected['name']}, {selected.get('admin1', '')}, {selected.get('country', '')} "
                    f"({selected['latitude']:.3f}, {selected['longitude']:.3f})")]:
                    frame = st.session_state.get("weather_history")
        except Exception as exc:
            st.error(f"Could not retrieve online data: {exc}")
else:
    uploaded = st.file_uploader("Upload daily weather data", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            raw = BytesIO(uploaded.getvalue())
            frame = pd.read_csv(raw) if uploaded.name.lower().endswith(".csv") else pd.read_excel(raw)
        except Exception as exc:
            st.error(f"Unable to read file: {exc}")

if frame is not None:
    st.subheader("Historical data")
    st.dataframe(frame.head(20), use_container_width=True)
    columns = list(frame.columns)
    date_index = next((i for i, x in enumerate(columns) if str(x).lower() in ("date", "time", "datetime")), 0)
    date_column = st.selectbox("Date column", columns, index=date_index)
    numeric_candidates = [x for x in columns if x != date_column]
    if not numeric_candidates:
        st.error("Your dataset needs at least one measurement column.")
        st.stop()
    preferred = "temperature_2m_mean"
    variable = st.selectbox("Variable to forecast", numeric_candidates,
                            index=numeric_candidates.index(preferred) if preferred in numeric_candidates else 0)
    st.info("This baseline forecasts one numeric variable at a time. Daily weather descriptions require a separate model.")
    if st.button("Train model and forecast", type="primary"):
        try:
            series = prepare_series(frame, date_column, variable)
            with st.spinner("Evaluating statistical models and generating forecasts..."):
                comparisons = None
                if model_choice.startswith("Automatic"):
                    winner, comparisons, errors = compare_models(series, order)
                    selected_model = winner["model"]
                    seasonal_order = winner["seasonal_order"]
                    metrics = {k: winner[k] for k in ("MAE", "RMSE", "Baseline MAE")}
                else:
                    selected_model = model_choice
                    seasonal_order = (1, 0, 0, 7) if model_choice.startswith("SARIMA") else (0, 0, 0, 0)
                    metrics = backtest(series, order, seasonal_order)
                predictions, aic = forecast(series, days, order, seasonal_order)
            st.session_state["arima_output"] = (series, predictions, metrics, aic, variable, selected_model, comparisons)
        except Exception as exc:
            st.error(f"Model could not be trained: {exc}")
    if "arima_output" in st.session_state:
        series, predictions, metrics, aic, label, selected_model, comparisons = st.session_state["arima_output"]
        st.subheader(f"Forecast: {label}")
        st.success(f"Model used: {selected_model}")
        if comparisons is not None:
            st.write("Model comparison (chronological holdout; lower MAE is better)")
            st.dataframe(pd.DataFrame(comparisons).drop(columns=["seasonal_order"]), use_container_width=True)
        cols = st.columns(4)
        cols[0].metric("Holdout MAE", f"{metrics['MAE']:.2f}")
        cols[1].metric("Holdout RMSE", f"{metrics['RMSE']:.2f}")
        cols[2].metric("Naive baseline MAE", f"{metrics['Baseline MAE']:.2f}")
        cols[3].metric("Model AIC", f"{aic:.1f}")
        fig = go.Figure()
        recent = series.iloc[-120:]
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, name="Historical"))
        fig.add_trace(go.Scatter(x=predictions["date"], y=predictions["predicted"], name="Predicted"))
        fig.add_trace(go.Scatter(x=predictions["date"], y=predictions["upper_95"],
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=predictions["date"], y=predictions["lower_95"],
                                 fill="tonexty", line=dict(width=0), name="95% interval"))
        fig.update_layout(xaxis_title="Date", yaxis_title=label, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(predictions, use_container_width=True)
        st.download_button("Download forecast CSV", predictions.to_csv(index=False),
                           "weather_forecast.csv", "text/csv")
        st.caption("Model selection uses one chronological holdout and may be optimistic. Weekly seasonality is not annual weather seasonality. Results are experimental.")
else:
    st.info("Choose a location and fetch online historical data, or upload a CSV/Excel dataset.")

# Phase 3: independent forecasts for several weather measurements.
if frame is not None:
    st.divider()
    st.subheader("Multi-variable forecasting")
    st.caption("Forecast multiple numerical weather variables. Each model is fitted independently; weather-condition labels are not yet predicted.")
    choices = [x for x in frame.columns if x != date_column and pd.to_numeric(frame[x], errors="coerce").notna().sum() >= 40]
    defaults = [x for x in ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"] if x in choices]
    selected_targets = st.multiselect(
        "Select variables", choices, default=defaults or choices[:1], max_selections=5)
    if st.button("Forecast selected variables", disabled=not selected_targets, type="primary"):
        try:
            with st.spinner("Training separate models for the selected variables..."):
                multi_predictions, multi_metrics, failures = forecast_many(
                    frame, date_column, selected_targets, days, order, model_choice)
            st.session_state["multi_output"] = (multi_predictions, multi_metrics, failures)
        except Exception as exc:
            st.error(f"Unable to generate multi-variable forecasts: {exc}")
    if "multi_output" in st.session_state:
        multi_predictions, multi_metrics, failures = st.session_state["multi_output"]
        st.dataframe(multi_metrics, use_container_width=True, hide_index=True)
        if failures:
            st.warning("Some targets could not be forecast: " + " | ".join(failures))
        for target in multi_predictions["variable"].unique():
            subset = multi_predictions[multi_predictions["variable"] == target]
            with st.expander(TARGETS.get(target, target), expanded=True):
                figure = go.Figure()
                figure.add_trace(go.Scatter(x=subset["date"], y=subset["predicted"],
                                            mode="lines+markers", name="Forecast"))
                figure.add_trace(go.Scatter(x=subset["date"], y=subset["upper_95"],
                                            line=dict(width=0), showlegend=False))
                figure.add_trace(go.Scatter(x=subset["date"], y=subset["lower_95"],
                                            fill="tonexty", line=dict(width=0), name="95% interval"))
                figure.update_layout(xaxis_title="Date", yaxis_title=TARGETS.get(target, target))
                st.plotly_chart(figure, use_container_width=True)
        st.download_button("Download multi-variable forecast CSV",
                           multi_predictions.to_csv(index=False),
                           "lameteo_multi_forecast.csv", "text/csv")
        st.caption("Precipitation and wind point forecasts are clipped to zero where negative. Model intervals are unadjusted.")

# Phase 4: lag-based weather condition classification.
if frame is not None:
    st.divider()
    st.subheader("Daily weather conditions")
    st.caption("Experimental classification of clear, cloudy, rainy and snowy days from historical WMO weather codes.")
    code_columns = [col for col in frame.columns if "weather_code" in str(col).lower()]
    if code_columns:
        code_column = st.selectbox("Historical weather code column", code_columns)
        if st.button("Predict daily conditions", type="primary"):
            try:
                with st.spinner("Training weather-condition classifier..."):
                    condition_forecast, condition_scores = predict_conditions(
                        frame, date_column, code_column, days)
                st.session_state["conditions_output"] = (
                    condition_forecast, condition_scores)
            except Exception as exc:
                st.error(f"Condition classifier could not run: {exc}")
        if "conditions_output" in st.session_state:
            condition_forecast, condition_scores = st.session_state["conditions_output"]
            a, b = st.columns(2)
            a.metric("Holdout accuracy", f"{condition_scores['holdout_accuracy']:.1%}")
            b.metric("Balanced accuracy", f"{condition_scores['holdout_balanced_accuracy']:.1%}")
            st.caption(f"Evaluated over {condition_scores['test_days']} historical days.")
            st.caption(f"Forecast begins after the final historical record: {condition_scores['history_end'].date()}.")
            st.dataframe(condition_forecast, hide_index=True, use_container_width=True)
            st.download_button("Download condition forecast CSV",
                               condition_forecast.to_csv(index=False),
                               "lameteo_conditions.csv", "text/csv")
            st.warning("These are simplified, recursive historical-pattern predictions, not official forecasts. Probabilities are uncalibrated and errors accumulate with horizon.")
    else:
        st.info("Condition forecasting requires a WMO weather_code column. Online historical data includes this column; uploaded files must provide it.")
