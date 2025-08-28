# planner.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd

# Keep this aligned with nlq_hf.DEFAULT_METRIC_ALIASES
_METRIC_ALIASES = {
    "spend": "total_landed_cost",
    "total_spend": "total_landed_cost",
    "landed_cost": "total_landed_cost",
    "cost": "total_landed_cost",
    "delay_days": "delay_days_vs_planned_eta",
    "avg_delay": "delay_days_vs_planned_eta",
    "on_time_percent": "on_time_percent",
    "on_time_%": "on_time_percent",
    "on-time %": "on_time_percent",
}

_DIMENSIONS = {"supplier", "lane", "mode", "status"}

def _resolve_metric(df: pd.DataFrame, metric: str | None) -> str | None:
    if not metric:
        # default to spend if present
        for alias in ["spend", "total_spend", "landed_cost", "cost"]:
            col = _METRIC_ALIASES.get(alias, alias)
            if col in df.columns:
                return col
        return None
    # map alias -> column if needed
    col = _METRIC_ALIASES.get(metric, metric)
    return col if col in df.columns else None

def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any] | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for k, v in filters.items():
        if k in out.columns:
            # support list-of-values OR scalar
            if isinstance(v, (list, tuple, set)):
                out = out[out[k].isin(list(v))]
            else:
                out = out[out[k] == v]
    return out

def _apply_time_range(df: pd.DataFrame, time_range: Dict[str, Any] | None) -> pd.DataFrame:
    if not time_range:
        return df
    # Implement your own timestamp column name if needed
    ts_col = None
    for candidate in ["created_at", "shipment_date", "eta", "updated_at", "date"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if not ts_col:
        return df
    ty = time_range.get("type")
    if ty == "last_n_days":
        n = int(time_range.get("n", 30))
        # Expect df[ts_col] is datetime64; if not, try to convert
        s = df[ts_col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=n)
        return df[s >= cutoff]
    return df

def _apply_limit(df: pd.DataFrame, limit: Dict[str, Any] | None) -> pd.DataFrame:
    """
    Enforce: {"dimension": "<supplier|lane|mode|status>", "n": N, "metric": "<col>"}
    Applies after filters/time range. Aggregation uses sum by default.
    """
    if not limit:
        return df
    dim = (limit.get("dimension") or "").strip()
    if dim not in _DIMENSIONS or dim not in df.columns:
        return df
    try:
        n = int(limit.get("n", 5))
        if n <= 0:
            return df
    except Exception:
        return df
    metric_col = _resolve_metric(df, limit.get("metric"))
    # If metric missing but user asked limit by status (counts), allow count-based top-N
    if metric_col is None:
        # fallback to count of rows per dim
        top_keys = (
            df.groupby(dim).size().sort_values(ascending=False).head(n).index
        )
        return df[df[dim].isin(top_keys)]
    # numeric aggregation by sum
    g = df.groupby(dim)[metric_col].sum().sort_values(ascending=False)
    top_keys = g.head(n).index
    return df[df[dim].isin(top_keys)]

def sanitize_plan(plan: Dict[str, Any], df_like: Any | None = None) -> str:
    """Compact representation for UI captions/logging."""
    try:
        import json
        return json.dumps(plan, ensure_ascii=False)[:800]
    except Exception:
        return str(plan)[:800]

def apply_llm_plan(data: pd.DataFrame, df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply the JSON plan produced by the LLM to the dataframe.
    This function focuses on filters, time_range, and the new 'limit' block.
    It is intentionally conservative to avoid surprises.
    """
    out = df

    # Time range first
    out = _apply_time_range(out, plan.get("time_range"))

    # Filters next
    out = _apply_filters(out, plan.get("filters"))

    # Limit by dimension
    out = _apply_limit(out, plan.get("limit"))

    # Optional: ordering/grouping can be added here if you already support it elsewhere.
    # We keep it minimal for "this work only".
    return out
