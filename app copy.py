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

    # --- NLQ UI (no sidebar, still uses LLM under the hood) ---
    st.divider()
    st.subheader("🔎 Ask with natural language")

    if "nl_prefill" in st.session_state:
        st.session_state["nl_prompt"] = st.session_state.pop("nl_prefill")

    q = st.text_input(
        "Ask about your shipments (e.g., 'top 5 suppliers by spend', 'show delayed air shipments from CN last 30 days').",
        key="nl_prompt",
        placeholder="Try: top 5 suppliers by spend",
    )

    suggestions = {
        "Quick starts": [
            "top 5 suppliers by spend last quarter",
            "average lead time by lane",
            "how many shipments last 30 days",
        ],
        "Shipment risk": [
            "show delayed air shipments from CN last 30 days",
            "top 10 lanes by delay",
            "show in-transit shipments US->IN last 14 days",
        ],
        "All data scope": [
            "top 5 suppliers by spend (all data)",
            "average delay by mode (all data)",
        ],
    }
    with st.expander("💡 Prompt suggestions (click to fill)", expanded=False):
        for section, items in suggestions.items():
            st.markdown(f"**{section}**")
            cols = st.columns(3)
            for i, s in enumerate(items):
                if cols[i % 3].button(s, key=f"suggest_{section}_{i}", use_container_width=True):
                    st.session_state["nl_prefill"] = s

    # Hidden LLM settings (no sidebar)
    use_llm = True  # keep LLM enabled but invisible
    model = get_secret("LLM_MODEL") or "openai/gpt-oss-20b:fireworks-ai"
    token = get_secret("HF_TOKEN")

    df_for_viz = df
    note: str | None = None

    if q:
        if use_llm and token:
            try:
                raw_plan = llm_plan_via_hf_router(q, token, model)
                df_for_viz = apply_llm_plan(data, df, raw_plan)
                note = f"LLM plan via {model}: {sanitize_plan(raw_plan, data)}"
            except Exception as e:
                df_for_viz, note = apply_prompt_filters(data, df, q)
                st.info(f"Used fallback parser (LLM error: {e})")
        else:
            if use_llm and not token:
                st.warning("Add HF_TOKEN to .streamlit/secrets.toml (or ENV) to enable the LLM parser.")
            df_for_viz, note = apply_prompt_filters(data, df, q)

    #if note:
        #st.caption(f"🧠 {note}")

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
