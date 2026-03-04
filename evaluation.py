import json
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
import pandas as pd
from embedding import TextProcessor, VectorStore

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH       = os.path.join(BASE_DIR, "scrapper", "output", "shl_individual_tests_enrich.json")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TRAIN_QUERIES_PATH = os.path.join(BASE_DIR, "dataset", "Train-Set.xlsx")
TOP_K_RETURN       = 10


def _load_assessments() -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_vector_store(assessments: Sequence[dict]) -> Tuple[TextProcessor, VectorStore]:
    processor = TextProcessor()
    texts = [processor.build_assessment_text(a) for a in assessments]
    embeddings = processor.get_embeddings(texts)
    store = VectorStore()
    store.create_index(embeddings, raw_texts=texts)
    return processor, store


def _init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")


def _load_labelled_queries(path: str = TRAIN_QUERIES_PATH) -> List[dict]:
    df = pd.read_excel(path)
    cols_lower = {c.lower(): c for c in df.columns}

    query_col      = cols_lower.get("query", df.columns[0])
    unique_queries = df[query_col].unique()
    url_col        = cols_lower.get("assessment_url", df.columns[1] if len(df.columns) > 1 else df.columns[0])

    labelled: List[dict] = []
    for query_text in unique_queries:
        relevant_urls = df[df[query_col] == query_text][url_col].tolist()
        labelled.append({"query": query_text, "relevant_urls": relevant_urls})

    return labelled


def _rerank_with_llm_eval(gemini_model, query: str, candidates: List[dict], top_k_return: int) -> List[dict]:
    if len(candidates) <= top_k_return:
        return candidates[:top_k_return]

    candidate_lines = []
    for i, c in enumerate(candidates):
        test_types = ", ".join(c.get("test_type", []))
        candidate_lines.append(
            f"{i}. Name: {c.get('name', '')} | "
            f"TestType: {test_types} | "
            f"Duration: {c.get('duration', 0)} mins | "
            f"Role: {c.get('role_summary', '')} | "
            f"Description: {c.get('description', '')[:400]}"
        )

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""
### ROLE
You are an SHL Assessment Recommendation Expert. Your task is to rank candidate assessments by their predictive power and relevance to a specific job requirement.
Predictive power means how well the assessment combination would evaluate success factors for this specific role.

### JOB REQUIREMENT
{query}

### CANDIDATES (Indices 0 to {len(candidates) - 1})
{candidates_text}

### EVALUATION FRAMEWORK
Apply these 6 criteria:
1. Skill Match: Direct alignment with required competencies.
2. Role Alignment: Match for seniority (Entry, Mid, Exec) and function.
   - Executive roles require broader leadership/strategic coverage.
   - Entry roles may prioritize foundational ability tests.
3. Cognitive Coverage: Prioritize reasoning/ability tests for analytical roles.
4. Behaviour Coverage: Prioritize personality/OPQ tests for people or leadership roles.
5. Technical Match: Prioritize domain-specific tests if tools/expertise are mentioned.
6. Battery Preference: Prefer a complementary combination (e.g., Cognitive + Behavioural)
   over redundant tests of the same construct, unless the role is highly specialized.

### RANKING INSTRUCTIONS
- Rank ALL candidates internally, then return only the top {top_k_return} indices in ranked order.
- Do not exclude assessments unless they are 0% relevant (place those at the very end).
- Prefer broader competency coverage over narrow filtering.
- Think step-by-step internally to ensure the Battery approach is applied.

Before finalizing:
- Ensure required cognitive and behavioural coverage is satisfied when applicable.
- Ensure no duplicate indices.

### OUTPUT CONSTRAINTS
- Return ONLY a valid JSON array of exactly {top_k_return} unique integers.
- The response MUST start with "[" and end with "]".
- DO NOT include markdown code blocks.
- DO NOT include backticks, preamble, or explanations.
- Example: [4, 0, 12, 7, 3]

### OUTPUT
"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        selected_indices = json.loads(raw)
        selected_indices = [
            i for i in selected_indices
            if isinstance(i, int) and 0 <= i < len(candidates)
        ]
    except Exception:
        selected_indices = []

    final = []
    seen  = set()

    for i in selected_indices:
        if i not in seen:
            final.append(candidates[i])
            seen.add(i)

    return final


def get_url_text(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if "view" in parts:
        return parts[-1]
    return parts[-1]


def evaluate_on_labelled_queries(
    top_k_retrieve: int = 50,
    top_k_final: int = 15,
    labelled_path: str = TRAIN_QUERIES_PATH,
) -> Dict[str, float]:

    labelled       = _load_labelled_queries(labelled_path)
    assessments    = _load_assessments()
    processor, store = _build_vector_store(assessments)
    gemini_model   = _init_gemini()

    dataset_text   = {get_url_text(a.get("url", "")) for a in assessments}
    all_queries    = [it.get("query", "").strip() for it in labelled]
    all_query_vecs = processor.get_embeddings(all_queries)

    retrieval_recalls = []
    final_recalls     = []
    processed_queries = 0

    for idx, it in enumerate(tqdm(labelled, desc="Evaluating labelled queries")):
        query    = all_queries[idx]
        query_vec = all_query_vecs[idx]
        all_relevant_urls = it.get("relevant_urls", [])

        relevant_text = {
            get_url_text(u) for u in all_relevant_urls
            if get_url_text(u) in dataset_text
        }

        if not query or not relevant_text:
            continue

        processed_queries += 1

        _, retrieved_indices = store.hybrid_search(
            query_embedding=query_vec,
            query_text=query,
            k=top_k_retrieve,
        )
        retrieved_indices = retrieved_indices[0]

        retrieved_text = {
            get_url_text(assessments[i].get("url", ""))
            for i in retrieved_indices if i >= 0
        }

        retrieval_recalls.append(len(retrieved_text.intersection(relevant_text)) / len(relevant_text))

        candidates = [assessments[i] for i in retrieved_indices if i >= 0]
        reranked   = _rerank_with_llm_eval(
            gemini_model=gemini_model,
            query=query,
            candidates=candidates,
            top_k_return=top_k_final,
        )[:TOP_K_RETURN] 
        

        final_text = {get_url_text(a.get("url", "")) for a in reranked}
        final_recalls.append(len(final_text.intersection(relevant_text)) / len(relevant_text))

    if processed_queries == 0:
        return {}

    return {
        f"retrieval_mean_recall@{top_k_retrieve}": float(np.mean(retrieval_recalls)),
        f"recommendation_mean_recall@{TOP_K_RETURN}": float(np.mean(final_recalls)),
    }


if __name__ == "__main__":
    try:
        print("\n=== Labelled train-data evaluation ===")
        metrics = evaluate_on_labelled_queries()
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"Error evaluating labelled queries: {e}")