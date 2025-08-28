from __future__ import annotations
import re
import json
import ast
import pandas as pd
from typing import Dict, Any

def extract_json(txt: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from model output.
    - Prefer <json>{...}</json> block
    - Strip code fences if present
    - Fallback to first {...} block
    - Try json.loads, then json5 (if installed), then ast.literal_eval
    """
    if not txt:
        return {}
    txt = txt.strip()

    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", txt, flags=re.S | re.I)
    if m:
        s = m.group(1)
    else:
        txt = txt.replace("```json", "```").replace("```JSON", "```")
        if "```" in txt:
            parts = [p for p in txt.split("```") if "{" in p]
            if parts:
                txt = parts[0]
        m = re.search(r"\{.*\}", txt, flags=re.S)
        s = m.group(0) if m else txt

    try:
        return json.loads(s)
    except Exception:
        try:
            import json5  # type: ignore
            return json5.loads(s)  # type: ignore
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            raise ValueError(f"Could not parse JSON: {e}. Snippet: {s[:300]}")

def resolve_relative_date_phrase(s: str) -> dict:
    """
    Map phrases like 'last week', 'last 30 days', 'yesterday', 'today' to absolute dates.
    Uses local today (no timezone).
    """
    s = (s or "").lower().strip()
    today = pd.Timestamp.today().normalize()

    def iso(d): return d.date().isoformat()

    if "last week" in s or "past week" in s or "last 7" in s:
        return {"start": iso(today - pd.Timedelta(days=7)), "end": iso(today)}
    if "last 30" in s or "past 30" in s or "last month" in s:
        return {"start": iso(today - pd.Timedelta(days=30)), "end": iso(today)}
    if "yesterday" in s:
        y = today - pd.Timedelta(days=1)
        return {"start": iso(y), "end": iso(today)}
    if "today" in s:
        return {"start": iso(today), "end": iso(today)}
    return {}
