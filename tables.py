from __future__ import annotations
import streamlit as st
import pandas as pd

def risky_shipments_table(df: pd.DataFrame) -> None:
    st.subheader("🚨 Risky Shipments")
    cols = [
        "shipment_id", "po_number", "supplier", "lane", "mode", "status",
        "po_date", "planned_eta", "actual_eta", "delay_days_vs_planned_eta", "total_landed_cost",
    ]
    view = df[df["risk_flag"] == 1].copy()
    if not view.empty:
        st.dataframe(
            view[cols].sort_values("delay_days_vs_planned_eta", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No risky shipments under the current filters.")

def download_filtered(df: pd.DataFrame) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", csv, "filtered_shipments.csv", "text/csv")

def data_dictionary_expander() -> None:
    with st.expander("Data Dictionary"):
        st.markdown(
            '''
- **po_value** = quantity × unit_price  
- **total_landed_cost** = po_value + freight_cost + duty_cost  
- **lead_time_days** = actual_ship_date − po_date  
- **transit_time_days** = actual_eta − actual_ship_date  
- **delay_days_vs_planned_eta** = actual_eta − planned_eta (days)  
- **on_time** = 1 if delivered on/before planned_eta else 0  
- **risk_flag** = 1 if (in-transit/delayed past planned_eta) or (delivered but >7 days late)  
'''
        )
