from __future__ import annotations
import pandas as pd
import numpy as np

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "po_value" not in df.columns:
        df["po_value"] = df["quantity"] * df["unit_price"]
    if "total_landed_cost" not in df.columns:
        df["total_landed_cost"] = df["po_value"] + df["freight_cost"] + df["duty_cost"]
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = (
            pd.to_datetime(df["actual_ship_date"]) - pd.to_datetime(df["po_date"])
        ).dt.days
    if "transit_time_days" not in df.columns:
        df["transit_time_days"] = (
            pd.to_datetime(df["actual_eta"]) - pd.to_datetime(df["actual_ship_date"])
        ).dt.days
    if "delay_days_vs_planned_eta" not in df.columns:
        df["delay_days_vs_planned_eta"] = (
            pd.to_datetime(df["actual_eta"]) - pd.to_datetime(df["planned_eta"])
        ).dt.days
    if "on_time" not in df.columns:
        df["on_time"] = np.where(
            (df["status"] == "Delivered")
            & (pd.to_datetime(df["actual_eta"]) <= pd.to_datetime(df["planned_eta"])),
            1, 0
        )
    if "risk_flag" not in df.columns:
        today = pd.Timestamp.utcnow().date()
        df["risk_flag"] = np.where(
            (
                (df["status"].isin(["In-Transit", "Delayed"]))
                & (pd.to_datetime(df["planned_eta"]).dt.date < today)
            ) | (
                (df["status"] == "Delivered")
                & (df["delay_days_vs_planned_eta"] > 7)
            ),
            1, 0
        )
    return df
