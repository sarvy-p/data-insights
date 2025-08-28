from __future__ import annotations
from typing import Dict
import streamlit as st
import pandas as pd
from constants import KPI_FORMATS

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    total_shipments = float(len(df))
    on_time_rate = (df["on_time"].mean() * 100.0) if total_shipments else 0.0
    avg_lead = float(df["lead_time_days"].mean()) if total_shipments else 0.0
    late_rate = (df["delay_days_vs_planned_eta"] > 0).mean() * 100.0 if total_shipments else 0.0
    total_spend = float(df["total_landed_cost"].sum())
    return dict(
        total_shipments=total_shipments,
        on_time_rate=on_time_rate,
        avg_lead=avg_lead,
        late_rate=late_rate,
        total_spend=total_spend,
    )

def render_kpis(kpis: Dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Shipments", KPI_FORMATS["total_shipments"].format(kpis["total_shipments"]))
    c2.metric("On-time Delivery %", KPI_FORMATS["on_time_rate"].format(kpis["on_time_rate"]))
    c3.metric("Avg Lead Time (days)", KPI_FORMATS["avg_lead"].format(kpis["avg_lead"]))
    c4.metric("Late Shipment Rate", KPI_FORMATS["late_rate"].format(kpis["late_rate"]))
    c5.metric("Total Spend (₹/$/€)", KPI_FORMATS["total_spend"].format(kpis["total_spend"]))
