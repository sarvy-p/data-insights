# nlq.py
from __future__ import annotations
import re
import pandas as pd
import streamlit as st
from typing import Tuple, Optional

# ---- Maps & vocab ----
_METRIC_MAP = {
    "spend": ("total_landed_cost", "sum"),
    "value": ("po_value", "sum"),
    "lead time": ("lead_time_days", "mean"),
    "delay": ("delay_days_vs_planned_eta", "mean"),
    "on-time": ("on_time", "mean"),
}
_DIM_MAP = {
    "supplier": "supplier",
    "suppliers": "supplier",
    "lane": "lane",
    "lanes": "lane",
    "mode": "mode",
    "modes": "mode",
    "origin": "origin_country",
    "origins": "origin_country",
    "status": "status",
    "statuses": "status",
}
_STATUS = {"delayed", "delivered", "in-transit", "in transit", "cancelled", "canceled"}
_MODE = {"air", "ocean", "road"}

# ---- Date parsers ----
def _last_n(text: str) -> Optional[pd.Timedelta]:
    m = re.search(r"last\s+(\d+)\s*(day|week|month|quarter|year)s?", text)
    if not m: return None
    n, unit = int(m.group(1)), m.group(2)
    days = {"day":1, "week":7, "month":30, "quarter":90, "year":365}[unit]
    return pd.Timedelta(days=n*days)

def _this_last_period(text: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if "this year" in text:   return (pd.Timestamp(today.year,1,1), today)
    if "last year" in text:   return (pd.Timestamp(today.year-1,1,1), pd.Timestamp(today.year-1,12,31))
    if "this quarter" in text:
        q = (today.month-1)//3 + 1
        start = pd.Timestamp(today.year, 3*(q-1)+1, 1); return (start, today)
    if "last quarter" in text:
        q = (today.month-1)//3 + 1; y = today.year; q -= 1
        if q == 0: q = 4; y -= 1
        start = pd.Timestamp(y, 3*(q-1)+1, 1); end = (start + pd.offsets.QuarterEnd(0)).normalize()
        return (start, min(end, today))
    if "this month" in text:  return (pd.Timestamp(today.year, today.month, 1), today)
    if "last month" in text:
        prev = today - pd.offsets.MonthBegin(1)
        start = pd.Timestamp(prev.year, prev.month, 1); end = (start + pd.offsets.MonthEnd(0)).normalize()
        return (start, min(end, today))
    return None

def _between_dates(text: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})", text)
    if not m: return None
    a, b = pd.to_datetime(m.group(1)), pd.to_datetime(m.group(2))
    start, end = (a, b) if a <= b else (b, a)
    return (start.normalize(), end.normalize())

def _build_date_mask(df: pd.DataFrame, text: str) -> Optional[pd.Series]:
    po = pd.to_datetime(df["po_date"], errors="coerce").dt.tz_localize(None)
    br = _between_dates(text)
    if br:
        start, end = br; return (po >= start) & (po < end + pd.Timedelta(days=1))
    td = _last_n(text)
    if td:
        end = po.max() if pd.notnull(po.max()) else pd.Timestamp.utcnow().tz_localize(None).normalize()
        start = end - td
        return (po >= start) & (po < end + pd.Timedelta(days=1))
    pr = _this_last_period(text)
    if pr:
        start, end = pr; return (po >= start) & (po < end + pd.Timedelta(days=1))
    return None

# ---- Text → filters ----
def _status_from_text(text: str) -> Optional[str]:
    for s in _STATUS:
        if s in text:
            return "In-Transit" if s in {"in-transit","in transit"} else s.capitalize()
    return None

def _mode_from_text(text: str) -> Optional[str]:
    for m in _MODE:
        if m in text: return m.capitalize()
    return None

def _origin_from_text(text: str) -> Optional[str]:
    m = re.search(r"\bfrom\s+([A-Z]{2})\b", text)
    return m.group(1) if m else None

def _supplier_from_text(text: str, df: pd.DataFrame) -> Optional[str]:
    for s in df["supplier"].dropna().unique():
        if s.lower() in text: return s
    return None

def _lane_from_text(text: str, df: pd.DataFrame) -> Optional[str]:
    # match explicit lane like "US->IN"
    m = re.search(r"\b([A-Z]{2})->([A-Z]{2})\b", text)
    if m:
        lane = m.group(0)
        if lane in set(df["lane"].astype(str)): return lane
    # else try literal lane text
    for l in df["lane"].dropna().unique():
        if str(l).lower() in text: return l
    return None

def _metric_from_text(text: str):
    for key, (col, agg) in _METRIC_MAP.items():
        if key in text: return col, agg, key
    if "cost" in text or "spend" in text: return "total_landed_cost","sum","spend"
    if "value" in text or "po value" in text: return "po_value","sum","value"
    if "lead time" in text: return "lead_time_days","mean","lead time"
    if "delay" in text or "late" in text: return "delay_days_vs_planned_eta","mean","delay"
    if "on time" in text or "on-time" in text or "otp" in text: return "on_time","mean","on-time"
    return None

def _dimension_from_text(text: str) -> Optional[str]:
    for k, v in _DIM_MAP.items():
        if re.search(rf"\b{k}\b", text): return v
    return None

# ==== NEW: turn a prompt into a DataFrame we can chart/KPI ====
def apply_prompt_filters(df_all: pd.DataFrame, df_filtered: pd.DataFrame, prompt: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df_for_visuals, note). We:
      - Start from 'all data' if the prompt includes that phrase, otherwise from the sidebar-filtered data.
      - Apply language-driven filters (status/mode/origin/supplier/lane/date).
      - If 'top N <dim> by <metric>' is requested, we reduce to only those top N groups.
    """
    text = (prompt or "").strip().lower()
    base = df_all if "all data" in text else df_filtered
    df = base.copy()
    applied = []

    # filters
    s = _status_from_text(text)
    if s: df = df[df["status"].str.lower() == s.lower()]; applied.append(f"status={s}")

    m = _mode_from_text(text)
    if m: df = df[df["mode"].str.lower() == m.lower()]; applied.append(f"mode={m}")

    o = _origin_from_text(text)
    if o: df = df[df["origin_country"].str.upper() == o.upper()]; applied.append(f"origin={o}")

    sup = _supplier_from_text(text, df_all)
    if sup: df = df[df["supplier"] == sup]; applied.append(f"supplier='{sup}'")

    lane = _lane_from_text(text, df_all)
    if lane: df = df[df["lane"] == lane]; applied.append(f"lane={lane}")

    dm = _build_date_mask(df, text)
    if dm is not None:
        df = df[dm]; applied.append("date range")

    # top N dim by metric -> keep only those groups so charts/KPIs reflect that slice
    mtop = re.search(r"top\s+(\d+)\s+([a-z\-]+)s?\s+by\s+([a-z\- ]+)", text)
    if mtop:
        n = int(mtop.group(1))
        dim_key = mtop.group(2).strip()
        met_key = mtop.group(3).strip()
        dim = _DIM_MAP.get(dim_key, _dimension_from_text(dim_key))
        metric = _metric_from_text(met_key) or _metric_from_text(text)
        if dim and metric and dim in df.columns:
            col, agg, key = metric
            g = df.groupby(dim, as_index=False)[col].agg(agg).sort_values(col, ascending=False).head(n)
            df = df[df[dim].isin(g[dim])]
            applied.append(f"top {n} {dim} by {key}")

    note = "Prompt on " + ("all data" if base is df_all else "filtered data")
    if applied:
        note += " • " + ", ".join(applied)
    return df, note

# (Optional) keep your earlier run_nl_query(...) if you still use it for ad-hoc tables/messages.
