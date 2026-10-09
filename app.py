"""Streamlit entry point for La Météo."""
from datetime import date, timedelta
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.forecasting import prepare_series, forecast, backtest, compare_models
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
