import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_data

st.title("🛠️ Build Your Own Index")

# Load Data
df, categories = load_data()
df.index = pd.to_datetime(df.index)
categories = categories.tolist()

selected_categories = st.multiselect("Select Categories", categories, default=categories[:3])

agg_method = st.selectbox("Aggregation Method", ["Sum", "Average"])
apply_smoothing = st.checkbox("Apply Moving Average Smoothing")
if apply_smoothing:
    ma_window = st.slider("Moving Average Window (months)", 1, 12, 3)

standardize_index = st.checkbox("Standardize Aggregated Index")

if selected_categories:
    df_selected = df[selected_categories]

    if agg_method == "Sum":
        index_series = df_selected.sum(axis=1)
    else:
        index_series = df_selected.mean(axis=1)

    if apply_smoothing:
        index_series = index_series.rolling(ma_window).mean()

    if standardize_index:
        index_series = (index_series - index_series.mean()) / index_series.std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(index_series.index, index_series)
    ax.set_title("Custom Index Over Time")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("Please select at least one category.")
