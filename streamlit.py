import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


# Point Streamlit to your deployed FastAPI backend
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://happy4040-shl-recommender.hf.space",
)


def check_api_health() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def call_recommend_api(query: str) -> Dict[str, Any]:
    payload = {"query": query}
    try:
        resp = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"Status {resp.status_code}: {resp.text}",
            }
        return {"ok": True, "data": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def build_results_dataframe(
    assessments: List[Dict[str, Any]], max_results: int
) -> pd.DataFrame:
    sliced = assessments[:max_results]
    rows = []
    for item in sliced:
        rows.append(
            {
                "name": item.get("name", ""), 
                "url": item.get("url", "")
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        layout="wide",
    )

    st.title("SHL Assessment Recommender")
    st.markdown(
        "Enter a hiring query or job description to get recommended SHL assessments. "
        "You can pass natural language, full JD text, or a JD URL."
    )

    with st.sidebar:
        st.subheader("Backend status")
        health = check_api_health()
        if health.get("ok"):
            st.success("API is up")
        else:
            st.error("API unreachable")
            if "error" in health:
                st.caption(health["error"])

        st.subheader("Settings")
        top_k = st.slider(
            "Number of recommendations to display",
            min_value=1,
            max_value=10,
            value=5,
        )

    input_mode = st.radio(
        "Input type",
        options=["Natural language query / JD text", "Job description URL"],
        index=0,
        help=(
            "Use the first option for free-text queries or full JD text. "
            "Use the URL option to paste a JD link; the backend will fetch and parse it."
        ),
    )

    if input_mode == "Job description URL":
        query_text = st.text_input(
            "Job description URL",
            placeholder="https://example.com/job-description",
        )
    else:
        query_text = st.text_area(
            "Query or job description",
            placeholder=(
                "E.g. 'I am hiring for Java developers who can also collaborate "
                "effectively with my business teams.'"
            ),
            height=160,
        )

    submitted = st.button("Get recommendations", type="primary")

    if submitted:
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            st.warning("Please enter a query, JD text, or URL.")
            return

        with st.spinner("Fetching recommendations..."):
            result = call_recommend_api(cleaned_query)

        if not result.get("ok"):
            st.error("Failed to get recommendations from the API.")
            if "error" in result:
                st.caption(result["error"])
            return

        data = result.get("data", {})
        recs = data.get("recommended_assessments", [])

        if not recs:
            st.info("No assessments were returned for this query.")
            return

        st.markdown("### Recommendations")
        st.caption(
            f"Showing up to {top_k} of {len(recs)} assessments returned by the API."
        )

        df = build_results_dataframe(recs, max_results=top_k)
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()