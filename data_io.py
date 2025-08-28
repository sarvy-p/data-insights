# data_io.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Iterable, Optional
from constants import EXPECTED_DATE_COLS

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Make all expected date columns tz-naive (drop UTC tz)
    for col in EXPECTED_DATE_COLS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.tz_localize(None)   # drop tz info
    return df

@st.cache_data
def load_sample() -> pd.DataFrame:
    df = pd.read_csv("sample_shipments.csv", parse_dates=EXPECTED_DATE_COLS, low_memory=False)
    return _normalize_dates(df)

def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=EXPECTED_DATE_COLS, low_memory=False)
    return _normalize_dates(df)

def validate_columns(df: pd.DataFrame, expected: Iterable[str]) -> Optional[str]:
    missing = [c for c in expected if c not in df.columns]
    return f"Missing required columns: {', '.join(missing)}" if missing else None
