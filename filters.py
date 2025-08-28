# filters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import streamlit as st
import pandas as pd
import datetime as _dt

@dataclass
class FilterState:
    date_range: Tuple[pd.Timestamp, pd.Timestamp]  # always valid (start, end)
    supplier: str
    lane: str
    mode: str
    status: str

# ---------- helpers ----------
def _data_min_max(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dmin = pd.to_datetime(df["po_date"], errors="coerce").min().normalize()
    dmax = pd.to_datetime(df["po_date"], errors="coerce").max().normalize()
    return dmin, dmax

def _default_year_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dmin, dmax = _data_min_max(df)
    start_candidate = dmax - pd.Timedelta(days=365)
    start = max(dmin, start_candidate)
    return start, dmax

def _defaults(df: pd.DataFrame) -> Dict[str, Any]:
    start, end = _default_year_range(df)
    return {
        "po_date_range": (start.date(), end.date()),  # widget-friendly dates
        "supplier": "All",
        "lane": "All",
        "mode": "All",
        "status": "All",
    }

def _ensure_model(df: pd.DataFrame) -> None:
    if "filters_model" not in st.session_state:
        st.session_state["filters_model"] = _defaults(df)
    if "_pending_clear" not in st.session_state:
        st.session_state["_pending_clear"] = False

def _consume_pending_clear(df: pd.DataFrame) -> None:
    if st.session_state.get("_pending_clear", False):
        st.session_state["filters_model"] = _defaults(df)
        st.session_state["_pending_clear"] = False  # consume trigger

def _coerce_range(val, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Make sure we always return a valid (start, end) Timestamp tuple."""
    def_start, def_end = _default_year_range(df)

    if val is None:
        return def_start, def_end

    if isinstance(val, (_dt.date, pd.Timestamp)):
        d = pd.to_datetime(val).normalize()
        return d, d

    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            return def_start, def_end
        if len(val) == 1:
            d0 = pd.to_datetime(val[0]) if val[0] is not None else def_start
            d0 = d0.normalize()
            return d0, d0
        d0 = pd.to_datetime(val[0]) if val[0] is not None else def_start
        d1 = pd.to_datetime(val[1]) if val[1] is not None else def_end
        d0 = d0.normalize(); d1 = d1.normalize()
        if d1 < d0:
            d0, d1 = d1, d0
        return d0, d1

    return def_start, def_end

# ---------- public API ----------
def sidebar_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Filters")

    # Set up model and consume any previous 'Remove filters' trigger BEFORE widgets
    _ensure_model(df)
    _consume_pending_clear(df)

    model: Dict[str, Any] = st.session_state["filters_model"]

    # Options
    suppliers = ["All"] + sorted(df["supplier"].dropna().unique().tolist())
    lanes     = ["All"] + sorted(df["lane"].dropna().unique().tolist())
    modes     = ["All"] + sorted(df["mode"].dropna().unique().tolist())
    statuses  = ["All"] + sorted(df["status"].dropna().unique().tolist())

    # Helper to get index safely
    def _idx(options, value):
        try:
            return options.index(value)
        except ValueError:
            return 0  # default to "All"

    # Widgets WITHOUT keys; values come from the model (so we can reset safely)
    date_input_val = st.sidebar.date_input("PO Date range", value=model["po_date_range"])
    supplier_val   = st.sidebar.selectbox("Supplier", suppliers, index=_idx(suppliers, model["supplier"]))
    lane_val       = st.sidebar.selectbox("Lane",     lanes,     index=_idx(lanes,     model["lane"]))
    mode_val       = st.sidebar.selectbox("Mode",     modes,     index=_idx(modes,     model["mode"]))
    status_val     = st.sidebar.selectbox("Status",   statuses,  index=_idx(statuses,  model["status"]))

    # Bottom full-width button (works on FIRST click)
    st.sidebar.markdown("---")
    if st.sidebar.button("Remove filters", use_container_width=True):
        # Set trigger; next run (which happens automatically) will reset BEFORE widgets render
        st.session_state["_pending_clear"] = True
        st.toast("Filters reset to last 1 year & All")

    # Build FilterState from coerced values (no mid-run writes to widget keys)
    start_ts, end_ts = _coerce_range(date_input_val, df)
    # Persist current choices into our model (safe — not widget keys)
    st.session_state["filters_model"] = {
        "po_date_range": (start_ts.date(), end_ts.date()),
        "supplier": supplier_val,
        "lane": lane_val,
        "mode": mode_val,
        "status": status_val,
    }

    return FilterState(
        date_range=(start_ts, end_ts),
        supplier=supplier_val,
        lane=lane_val,
        mode=mode_val,
        status=status_val,
    )

def apply_filters(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    """Always apply date (default = last 1 year unless user changed it), then selects."""
    end_plus = f.date_range[1] + pd.Timedelta(days=1)

    # tz-naive for safe comparisons
    po_dt = pd.to_datetime(df["po_date"], errors="coerce").dt.tz_localize(None)

    out = df[(po_dt >= f.date_range[0]) & (po_dt < end_plus)]
    if f.supplier != "All":
        out = out[out["supplier"] == f.supplier]
    if f.lane != "All":
        out = out[out["lane"] == f.lane]
    if f.mode != "All":
        out = out[out["mode"] == f.mode]
    if f.status != "All":
        out = out[out["status"] == f.status]
    return out
