# app.py (final with bigger highlighted NLQ box + suggestion buttons)
from __future__ import annotations
import streamlit as st

from constants import APP_TITLE, EXPECTED_COLS
from data_io import load_sample, load_uploaded, validate_columns
from features import derive_features
from filters import sidebar_filters, apply_filters
from kpis import compute_kpis, render_kpis
from charts import render_charts
from tables import risky_shipments_table, download_filtered, data_dictionary_expander
from nlq import apply_prompt_filters

from app_secrets import get_secret
from llm_router import llm_plan_via_hf_router
from planner import apply_llm_plan, sanitize_plan
from ui import header, data_source_picker, footer_description

# enrichment helper
from nlq_hf import extract_top_limit_from_text


def main() -> None:
    header(APP_TITLE)

    # Data load + filters
    data = data_source_picker(
        EXPECTED_COLS,
        load_sample,
        load_uploaded,
        validate_columns,
        derive_features,
    )
    filters = sidebar_filters(data)
    df = apply_filters(data, filters)

    # --- NLQ UI (bigger + highlighted) ---
    st.divider()
    st.subheader("🔎 Ask with natural language")

    # One-time CSS to style the NLQ area
    st.markdown(
        """
<style>
/* Wrap card around the text area */
div[data-testid="stTextArea"]{
  border: 2px solid #f59e0b;      /* amber-500 */
  background: #fff8e1;            /* amber-50 */
  border-radius: 14px;
  padding: 10px 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* Make the textarea text larger and comfier */
div[data-testid="stTextArea"] textarea {
  font-size: 18px !important;
  line-height: 1.5 !important;
  padding: 12px !important;
}

/* Tweak the label */
div[data-testid="stTextArea"] label p {
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  margin-bottom: 6px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # Support pre-filling the input from suggestion buttons
    if "nl_prefill" in st.session_state:
        st.session_state["nl_prompt"] = st.session_state.pop("nl_prefill")

    # Multiline text area (bigger than text_input)
    q = st.text_area(
        "Ask about your shipments",
        key="nl_prompt",
        placeholder="Try: Show risky shipments of Foxtrot Fasteners for lane US → IN with mode Ocean",
        height=120,
    )

    # --- Suggestion Box (click to prefill) ---
    st.caption("💡 Suggestions (click to prefill)")
    suggestions = [
        "Show risky shipments of Delta Textiles",
        "Show risky shipments for lane SG → IN with mode Road",
        "Show risky shipments of Echo Electronics for lane VN → IN with mode Ocean",
        "Show risky shipments of Crescent Metals for lane CN → IN with mode Ocean",
    ]

    # Two-per-row layout so buttons don’t get cramped
    for i in range(0, len(suggestions), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(suggestions):
                if cols[j].button(suggestions[idx], key=f"suggestion_{idx}"):
                    st.session_state["nl_prefill"] = suggestions[idx]
                    st.rerun()

    # Hidden LLM settings (no sidebar)
    use_llm = True  # keep LLM enabled but invisible
    model = get_secret("LLM_MODEL") or "openai/gpt-oss-20b:fireworks-ai"
    token = get_secret("HF_TOKEN")

    df_for_viz = df
    note: str | None = None

    if q:
        if use_llm and token:
            try:
                # Pass the available columns to improve planning quality
                raw_plan = llm_plan_via_hf_router(q, token, model, available_columns=list(df.columns))

                # ---------- Enrichment step (guarantee keys & fill gaps) ----------
                # Ensure required keys exist
                for k in ["time_range", "filters", "group_by", "order_by", "limit", "select"]:
                    raw_plan.setdefault(k, None)
                if raw_plan["filters"] is None:
                    raw_plan["filters"] = {}

                ql = q.lower()

                # Enrich time_range if missing
                if raw_plan["time_range"] is None:
                    if "last 30 days" in ql:
                        raw_plan["time_range"] = {"type": "last_n_days", "n": 30}
                    elif "last 14 days" in ql:
                        raw_plan["time_range"] = {"type": "last_n_days", "n": 14}
                    elif "last quarter" in ql:
                        raw_plan["time_range"] = {"type": "last_n_days", "n": 90}

                # Enrich limit if missing
                if raw_plan["limit"] is None:
                    top_info = extract_top_limit_from_text(q)
                    if top_info:
                        raw_plan["limit"] = {
                            "dimension": top_info["dimension"],
                            "n": int(top_info["n"]),
                            "metric": "total_landed_cost",
                        }

                # Enrich order_by if missing
                if raw_plan["order_by"] is None:
                    if "spend" in ql:
                        raw_plan["order_by"] = {"metric": "total_landed_cost", "direction": "desc"}
                    elif "delay" in ql:
                        raw_plan["order_by"] = {"metric": "delay_days_vs_planned_eta", "direction": "desc"}
                    elif "on-time" in ql or "on time" in ql:
                        raw_plan["order_by"] = {"metric": "on_time_percent", "direction": "desc"}

                # If user mentions "risky shipments" and filters lack status, add it
                if "risky" in ql and "status" not in (raw_plan["filters"] or {}):
                    raw_plan["filters"]["status"] = "Delayed"
                # ---------- End enrichment ----------

                # Optional debug:
                # st.json(raw_plan)

                df_for_viz = apply_llm_plan(data, df, raw_plan)
                note = f"LLM plan via {model}: {sanitize_plan(raw_plan, data)}"
            except Exception as e:
                df_for_viz, note = apply_prompt_filters(data, df, q)
                st.info(f"Used fallback parser (LLM error: {e})")
        else:
            if use_llm and not token:
                st.warning("Add HF_TOKEN to .streamlit/secrets.toml (or ENV) to enable the LLM parser.")
            df_for_viz, note = apply_prompt_filters(data, df, q)

    # if note:
    #     st.caption(f"🧠 {note}")

    # KPIs
    st.divider()
    kpis = compute_kpis(df_for_viz)
    render_kpis(kpis)

    # Table + download
    st.divider()
    risky_shipments_table(df_for_viz)
    download_filtered(df_for_viz)

    # Charts
    st.divider()
    render_charts(df_for_viz)

    # Data dictionary + footer
    st.divider()
    data_dictionary_expander()
    footer_description()


if __name__ == "__main__":
    main()
