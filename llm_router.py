# llm_router.py (uses heuristic fallback to build plan when JSON missing)
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

import requests

from nlq_hf import make_hf_messages, build_heuristic_plan

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
COMPL_URL = "https://router.huggingface.co/v1/completions"


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    s = text
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return s


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _coerce_to_json_dict_or_none(content: str) -> Optional[Dict[str, Any]]:
    # Try fast path
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    txt = _strip_code_fences(content)
    txt = _normalize_quotes(txt)
    candidate = _extract_json_object(txt)
    candidate = _remove_trailing_commas(candidate)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_content_from_chat(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices:
        return None
    first = choices[0]
    if isinstance(first.get("text"), str) and first["text"].strip():
        return first["text"]
    msg = first.get("message")
    if isinstance(msg, dict):
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            return msg["content"]
        if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip():
            return msg["reasoning_content"]
    if isinstance(first.get("generated_text"), str) and first["generated_text"].strip():
        return first["generated_text"]
    return None


def _extract_content_from_completions(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices:
        return None
    first = choices[0]
    if isinstance(first.get("text"), str) and first["text"].strip():
        return first["text"]
    return None


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"[{role.upper()}]\n{content}\n")
    lines.append("Return ONLY the JSON plan.")
    return "\n".join(lines)


def llm_plan_via_hf_router(
    query: str,
    token: str,
    model: str,
    available_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Try chat first; if model not chat, fall back to completions.
    If the model doesn't return JSON at all, build a heuristic plan so the app still works.
    """
    messages = make_hf_messages(query, available_columns or [])
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # 1) chat
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=90)

    if resp.status_code == 400:
        try:
            err = resp.json()
            err_str = json.dumps(err)
        except Exception:
            err_str = resp.text or ""
        if "not a chat model" in err_str.lower() or "model_not_supported" in err_str.lower() or "response_format" in err_str.lower():
            # 2) completions
            prompt = _messages_to_prompt(messages)
            payload2 = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 768,
                "temperature": 0.0,
                "stream": False,
            }
            resp2 = requests.post(COMPL_URL, headers=headers, json=payload2, timeout=90)
            if resp2.status_code >= 400:
                # Last resort: heuristic plan
                return build_heuristic_plan(query)
            data2 = resp2.json()
            content2 = _extract_content_from_completions(data2)
            if not content2:
                return build_heuristic_plan(query)
            plan = _coerce_to_json_dict_or_none(content2)
            return plan if plan is not None else build_heuristic_plan(query)

    if resp.status_code >= 400:
        # Heuristic plan on hard errors
        return build_heuristic_plan(query)

    data = resp.json()
    content = _extract_content_from_chat(data)
    if not content:
        # Try completions fetch, else heuristic
        prompt = _messages_to_prompt(messages)
        resp2 = requests.post(COMPL_URL, headers=headers, json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 768,
            "temperature": 0.0,
            "stream": False,
        }, timeout=90)
        if resp2.status_code < 400:
            data2 = resp2.json()
            content2 = _extract_content_from_completions(data2)
            if content2:
                plan2 = _coerce_to_json_dict_or_none(content2)
                return plan2 if plan2 is not None else build_heuristic_plan(query)
        return build_heuristic_plan(query)

    plan = _coerce_to_json_dict_or_none(content)
    return plan if plan is not None else build_heuristic_plan(query)
