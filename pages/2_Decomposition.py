import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data

st.title("📊 Statistics")

# Load Data
df, categories = load_data()
df.index = pd.to_datetime(df.index)
categories = categories.tolist()

# Select categories
selected_categories = st.multiselect("Select Categories", categories, default=categories[:3])

# Select date range
min_date, max_date = df.index.min(), df.index.max()
date_range = st.date_input("Select Date Range", [min_date, max_date])

# Standardize
standardize = st.checkbox("Standardize Data")

if selected_categories:
    df_filtered = df[selected_categories].copy()
    if len(date_range) == 2:
        df_filtered = df_filtered.loc[date_range[0]:date_range[1]]

    if standardize:
        df_filtered = (df_filtered - df_filtered.mean()) / df_filtered.std()

    # Line Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_filtered.columns:
        ax.plot(df_filtered.index, df_filtered[col], label=col)
    ax.set_title("Publications Over Time")
    ax.legend()
    st.pyplot(fig)

    # Summary stats
    st.write("**Summary Statistics:**")
    st.dataframe(df_filtered.describe())

    # Correlation
    st.write("**Correlation Heatmap:**")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("Please select at least one category.")
