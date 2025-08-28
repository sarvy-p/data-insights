# nlq_hf.py (LLM-led fuzzy entity mapping + richer instructions for statuses)
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Union

# Number words -> digits for robust "top N" parsing
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}

# Metric & dimension aliases
DEFAULT_METRIC_ALIASES = {
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

DIMENSION_ALIASES = {
    "supplier": ["supplier", "suppliers", "vendor", "vendors"],
    "lane": ["lane", "lanes"],
    "mode": ["mode", "modes"],
    "status": ["status", "statuses"],
}

STATUS_SYNONYMS = {
    # normalize many "risky"/"delay" ways to the same label your dataset uses
    "Delayed": [
        "risky", "at risk", "risk", "behind schedule", "late", "delayed", "overdue",
        "running late", "lagging", "slipped"
    ]
}


def _normalize_number_words_to_digits(text: str) -> str:
    def repl(m):
        w = m.group(0).lower()
        return str(_WORD_TO_NUM.get(w, w))
    return re.sub(r"\b(" + "|".join(_WORD_TO_NUM.keys()) + r")\b", repl, text, flags=re.IGNORECASE)


def _available_col_hints(available_columns: List[str]) -> str:
    sample_cols = ", ".join(sorted(available_columns)[:40])
    return f"Available columns include: {sample_cols}." if sample_cols else ""


def _dimension_hint() -> str:
    parts = []
    for dim, aliases in DIMENSION_ALIASES.items():
        parts.append(f"{dim}: {', '.join(aliases)}")
    return "Dimension aliases (map user terms to dataframe columns): " + " | ".join(parts) + "."


def _metric_hint() -> str:
    items = [f"{k} -> {v}" for k, v in DEFAULT_METRIC_ALIASES.items()]
    return "Metric aliases (map user metric names to dataframe columns): " + " | ".join(items) + "."


def _value_hints(value_hints: Dict[str, List[str]] | None) -> str:
    """
    value_hints may include lists of known values to help the LLM fuzzy-map names:
      {
        "suppliers": ["Foxtrot Fasteners", "Acme Co", ...],
        "modes": ["Air", "Ocean", "Truck", "Rail"],
        "statuses": ["Delayed", "In-Transit", "Delivered"],
        "lanes": ["US->IN", "CN->US", ...]
      }
    """
    if not value_hints:
        return ""
    lines = []
    for k in ("suppliers", "modes", "statuses", "lanes"):
        vals = value_hints.get(k)
        if vals:
            preview = ", ".join(map(str, vals[:60]))
            lines.append(f"Known {k}: {preview}")
    return "\n".join(lines)


PLANNER_SYSTEM_PROMPT = """
You are a planner that converts a user's shipment analytics request into a single JSON object called a "plan".
Your output MUST be valid JSON and MUST contain the keys below (use null when a key does not apply). Do not output any text outside JSON.

REQUIRED KEYS (always include all keys; use null if not applicable):
- "time_range": object or null
- "filters": object (can be {})
- "group_by": string or null
- "order_by": object or null   // {"metric": "<col>", "direction": "desc|asc"}
- "limit": object or null      // {"dimension": "supplier|lane|mode|status", "n": <int>, "metric": "<col>"}
- "select": array or null

ENTITY & SYNONYM MAPPING (IMPORTANT)
- If the user mentions a SUPPLIER by name (even misspelled), map it to the closest known supplier value using fuzzy matching.
  If a "Known suppliers" list is provided in hints, choose the closest item from that list.
- Map synonyms for risky/delayed shipments to a status filter. Words like: "risky", "at risk", "behind schedule", "late", "delayed", "overdue" => {"status":"Delayed"}.
- For modes, map "air/ocean/truck/rail" to the "mode" filter when present.
- For lanes, detect patterns like "US->IN" (origin->destination) and set {"lane":"US->IN"}.

TIME RANGES
- "last 30 days" -> {"time_range":{"type":"last_n_days","n":30}}, similarly for 14, 90 (last quarter ~= 90 days).

ORDERING & LIMITS
- Recognize "top/limit N <dimension> by <metric>". N can be digits or number words.
- Emit: {"limit":{"dimension":"supplier|lane|mode|status","n":N,"metric":"<col>"}}.
- Default metric for "by spend" is the spend column.
- Apply "limit" after applying filters/time_range.
- If the query implies ranking by a dimension (e.g., "top suppliers by spend"), set "group_by" to that dimension and "order_by" to the chosen metric (desc by default).

OUTPUT SHAPE (EXAMPLE)
{
  "time_range": { "type": "last_n_days", "n": 30 },
  "filters": { "status": "Delayed", "supplier": "Foxtrot Fasteners" },
  "group_by": "supplier",
  "order_by": { "metric": "total_landed_cost", "direction": "desc" },
  "limit": { "dimension": "supplier", "n": 2, "metric": "total_landed_cost" },
  "select": null
}
"""


# Few-shot examples focusing on fuzzy supplier names & risky synonyms
FEW_SHOTS = [
    {
        "user": "show risky shipments of supplier foxtrot fastners last 30 days",
        "assistant": {
            "time_range": { "type": "last_n_days", "n": 30 },
            "filters": { "status": "Delayed", "supplier": "Foxtrot Fasteners" },
            "group_by": None,
            "order_by": None,
            "limit": None,
            "select": None
        }
    },
    {
        "user": "delayed shipments for vendor acme co",
        "assistant": {
            "time_range": None,
            "filters": { "status": "Delayed", "supplier": "Acme Co" },
            "group_by": None,
            "order_by": None,
            "limit": None,
            "select": None
        }
    },
    {
        "user": "top two suppliers by spend last quarter",
        "assistant": {
            "time_range": { "type": "last_n_days", "n": 90 },
            "filters": {},
            "group_by": "supplier",
            "order_by": { "metric": "total_landed_cost", "direction": "desc" },
            "limit": { "dimension": "supplier", "n": 2, "metric": "total_landed_cost" },
            "select": None
        }
    }
]


def make_hf_messages(
    query: str,
    available_columns_or_hints: Union[List[str], Dict[str, List[str]], None] = None
) -> list[dict[str, str]]:
    """
    Backward-compatible signature:
    - If a list is passed, it's treated as the available COLUMNS (old behavior).
    - If a dict is passed, it's treated as VALUE HINTS for fuzzy-mapping (e.g., suppliers list).
    """
    query_norm = _normalize_number_words_to_digits(query)

    col_hints = ""
    value_hints_txt = ""
    if isinstance(available_columns_or_hints, list):
        col_hints = _available_col_hints(available_columns_or_hints or [])
    elif isinstance(available_columns_or_hints, dict):
        value_hints_txt = _value_hints(available_columns_or_hints)

    hints = "\n".join(
        h for h in [
            col_hints,
            value_hints_txt,
            _dimension_hint(),
            _metric_hint()
        ] if h
    )

    system = PLANNER_SYSTEM_PROMPT + ("\n\n" + hints if hints else "")

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # Few-shot exemplars (assistant replies are raw JSON only)
    import json as _json
    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": _json.dumps(ex["assistant"])})

    messages.append({
        "role": "user",
        "content": f"User question: {query_norm}\nReturn ONLY the JSON plan with all required keys."
    })
    return messages


# Heuristic helpers (kept for hybrid fallback in router)
def extract_top_limit_from_text(query: str) -> Optional[Dict[str, Any]]:
    q = _normalize_number_words_to_digits(query.lower())
    patterns = []
    for dim, aliases in DIMENSION_ALIASES.items():
        alias_pat = r"(?:%s)" % "|".join(map(re.escape, aliases))
        patterns.append((dim, re.compile(rf"\b(?:top|limit)\s+(\d+)\s+{alias_pat}\b")))
        patterns.append((dim, re.compile(rf"\b(?:top|limit)\s+(\d+)\s+{alias_pat}s\b")))
    for dim, pat in patterns:
        m = pat.search(q)
        if m:
            try:
                n = int(m.group(1))
                return {"dimension": dim, "n": n}
            except ValueError:
                pass
    return None


def build_heuristic_plan(query: str) -> Dict[str, Any]:
    """
    Construct a best-effort plan dict from the raw text query (used as router fallback).
    """
    ql = query.lower()

    # time_range
    time_range = None
    if "last 30 days" in ql:
        time_range = {"type": "last_n_days", "n": 30}
    elif "last 14 days" in ql:
        time_range = {"type": "last_n_days", "n": 14}
    elif "last quarter" in ql:
        time_range = {"type": "last_n_days", "n": 90}

    # filters (includes risky/delayed synonyms)
    filters: Dict[str, Any] = {}
    if any(w in ql for w in STATUS_SYNONYMS["Delayed"]):
        filters["status"] = "Delayed"

    # naive supplier capture (still let LLM correct via few-shots when used)
    m = re.search(r"(?:supplier|vendor)\s+([a-z0-9\-\s&\.]+)", ql)
    if m:
        supplier_name = m.group(1).strip().title()
        filters["supplier"] = supplier_name

    # limit
    limit = None
    top_info = extract_top_limit_from_text(query)
    if top_info:
        metric = "total_landed_cost"
        if "delay" in ql:
            metric = "delay_days_vs_planned_eta"
        elif "on-time" in ql or "on time" in ql:
            metric = "on_time_percent"
        limit = {"dimension": top_info["dimension"], "n": int(top_info["n"]), "metric": metric}

    # group_by & order_by
    group_by = None
    order_by = None
    if limit:
        group_by = limit["dimension"]
        order_by = {"metric": limit["metric"], "direction": "desc"}
    else:
        if "by spend" in ql:
            order_by = {"metric": "total_landed_cost", "direction": "desc"}
        elif "by delay" in ql:
            order_by = {"metric": "delay_days_vs_planned_eta", "direction": "desc"}
        elif "by on-time" in ql or "by on time" in ql:
            order_by = {"metric": "on_time_percent", "direction": "desc"}

    return {
        "time_range": time_range,
        "filters": filters,
        "group_by": group_by,
        "order_by": order_by,
        "limit": limit,
        "select": None,
    }
