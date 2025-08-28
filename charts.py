from __future__ import annotations
import pandas as pd
import altair as alt
import streamlit as st

def spend_by_supplier_chart(df: pd.DataFrame) -> alt.Chart:
    data = (
        df.groupby("supplier", as_index=False)["total_landed_cost"]
        .sum()
        .sort_values("total_landed_cost", ascending=False)
        .head(12)
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("total_landed_cost:Q", title="Total Landed Cost"),
            y=alt.Y("supplier:N", sort="-x", title="Supplier"),
            tooltip=["supplier", "total_landed_cost"],
        )
    )

def otp_by_supplier_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.groupby("supplier", as_index=False)["on_time"].mean()
    data["on_time_pct"] = data["on_time"] * 100.0
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("on_time_pct:Q", title="On-time %"),
            y=alt.Y("supplier:N", sort="-x", title="Supplier"),
            tooltip=["supplier", "on_time_pct"],
        )
    )

def lead_time_hist_chart(df: pd.DataFrame) -> alt.Chart:
    base = df.dropna(subset=["lead_time_days"])
    return (
        alt.Chart(base)
        .mark_bar()
        .encode(
            x=alt.X("lead_time_days:Q", bin=alt.Bin(maxbins=30), title="Lead Time (days)"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Shipments")],
        )
    )

def lane_delay_chart(df: pd.DataFrame) -> alt.Chart:
    data = (
        df.groupby("lane", as_index=False)["delay_days_vs_planned_eta"]
        .mean()
        .sort_values("delay_days_vs_planned_eta", ascending=False)
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("delay_days_vs_planned_eta:Q", title="Avg Delay vs Planned ETA (days)"),
            y=alt.Y("lane:N", sort="-x", title="Lane"),
            tooltip=["lane", "delay_days_vs_planned_eta"],
        )
    )

def render_charts(df: pd.DataFrame) -> None:
    l, r = st.columns(2)
    with l:
        st.subheader("Spend by Supplier")
        st.altair_chart(spend_by_supplier_chart(df), use_container_width=True)
    with r:
        st.subheader("On-time % by Supplier")
        st.altair_chart(otp_by_supplier_chart(df), use_container_width=True)

    l2, r2 = st.columns(2)
    with l2:
        st.subheader("Lead Time Distribution (days)")
        st.altair_chart(lead_time_hist_chart(df), use_container_width=True)
    with r2:
        st.subheader("Lane Delay (avg days vs planned ETA)")
        st.altair_chart(lane_delay_chart(df), use_container_width=True)
