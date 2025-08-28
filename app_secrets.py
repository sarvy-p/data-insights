from __future__ import annotations
import os
import streamlit as st

def get_secret(key: str) -> str | None:
    v = os.environ.get(key)
    if v:
        return v
    try:
        return st.secrets.get(key)  # type: ignore[attr-defined]
    except Exception:
        return None
