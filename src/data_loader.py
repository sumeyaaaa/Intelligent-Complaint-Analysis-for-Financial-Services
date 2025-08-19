import streamlit as st
import pandas as pd
from datetime import date

@st.cache_data
def load_and_preprocess_data(data_path):
    """Loads and caches the complaint data."""
    try:
        df = pd.read_csv(data_path)
        df["Date received"] = pd.to_datetime(df["Date received"])
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {data_path}. Please check your project structure.")
        return pd.DataFrame()

def get_filtered_data(df_full, start_date, end_date):
    """Filters the DataFrame based on the selected date range."""
    return df_full[
        (df_full["Date received"] >= pd.to_datetime(start_date)) &
        (df_full["Date received"] <= pd.to_datetime(end_date))
    ]