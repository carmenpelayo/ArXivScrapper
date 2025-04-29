import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

st.title("📈 Forecasting")

# Load and prepare data
df = pd.read_excel("arxiv_monthly_publications.xlsx", index_col=0)
df.index = pd.to_datetime(df.index)

# Select Category
categories = df.columns.tolist()
selected_category = st.selectbox("Select Category for Forecast", categories)

# Forecast Horizon
future_months = st.slider("Months to Forecast", 3, 48, 12)

if selected_category:
    df_prophet = df[[selected_category]].reset_index()
    df_prophet.columns = ["ds", "y"]

    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=future_months, freq='MS')
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_prophet["ds"], df_prophet["y"], label="Actual Data", color="2DCCCD")
        ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", linestyle="--", color="004481")
        ax.set_title(f"Forecast for {selected_category}")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
