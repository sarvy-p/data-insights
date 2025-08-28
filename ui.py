from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Callable, Optional

def header(app_title: str) -> None:
    st.set_page_config(page_title="Procurement & Shipment Insights", layout="wide")
    st.title(app_title)

def data_source_picker(
    expected_cols,
    load_sample: Callable[[], pd.DataFrame],
    load_uploaded: Callable[[object], pd.DataFrame],
    validate_columns: Callable[[pd.DataFrame, list], Optional[str]],
    derive_features: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    st.sidebar.header("Data")
    use_sample = st.sidebar.checkbox("Use sample data", value=True)
    uploaded = st.sidebar.file_uploader("Upload shipment CSV", type=["csv"])

    if use_sample:
        df = load_sample()
    elif uploaded:
        df = load_uploaded(uploaded)
    else:
        st.info("Upload a file or check 'Use sample data' to get started.")
        st.stop()

    err = validate_columns(df, expected_cols)
    if err:
        st.error(err)
        st.stop()

    return derive_features(df)

def footer_description() -> None:
    with st.expander("ℹ️ Note: About this app and expected data format", expanded=False):
        st.markdown(
            """
            **Quick insights** on suppliers, lanes, on-time delivery, lead times, and spend.  
            Upload your CSV or use the sample dataset.  

            **Expected columns:**  
            `po_number, supplier, origin_country, destination_country, lane, mode, incoterm, 
            po_date, planned_ship_date, actual_ship_date, planned_eta, actual_eta, status, 
            quantity, unit_price, freight_cost, duty_cost, shipment_id`
            """
        )
