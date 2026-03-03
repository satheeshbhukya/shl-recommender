import json
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
import pandas as pd
from embedding import TextProcessor, VectorStore
from expand_query import expand_query


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "scrapper", "output", "shl_individual_tests.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



TRAIN_QUERIES_PATH = os.path.join(BASE_DIR, "dataset", "Train-Set.xlsx")


def _load_assessments() -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_vector_store(assessments: Sequence[dict]) -> Tuple[TextProcessor, VectorStore]:
    processor = TextProcessor()
    texts = [processor.build_assessment_text(a) for a in assessments]
    embeddings = processor.get_embeddings(texts)
    store = VectorStore()
    store.create_index(embeddings, raw_texts=texts)  # FIX: pass raw_texts to enable BM25
    return processor, store


def _init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")


def _hit_rate_at_k(ranks: List[int], k: int) -> float:
    hits = sum(1 for r in ranks if 0 <= r < k)
    return hits / len(ranks) if ranks else 0.0


def _mrr_at_k(ranks: List[int], k: int) -> float:
    total = 0.0
    for r in ranks:
        if 0 <= r < k:
            total += 1.0 / (r + 1)
    return total / len(ranks) if ranks else 0.0


def _load_labelled_queries(path: str = TRAIN_QUERIES_PATH) -> List[dict]:

    df = pd.read_excel(path)
    cols_lower = {c.lower(): c for c in df.columns}

    query_col = cols_lower.get("query", df.columns[0])
    unique_queries = df[query_col].unique()
    url_col = cols_lower.get("assessment_url", df.columns[1] if len(df.columns) > 1 else df.columns[0])

    labelled: List[dict] = []
    for query_text in unique_queries:
        relevant_urls = df[df[query_col] == query_text][url_col].tolist()

        labelled.append(
            {
                "query": query_text,
                "relevant_urls": relevant_urls,
            }
        )

    return labelled 

def _retrieve_with_expansion(query: str, processor: TextProcessor, store: VectorStore, 
    gemini_model, top_k: int, rrf_k: int = 60,) -> List[int]: 

    sub_queries = expand_query(query, gemini_model)
    rrf_scores: Dict[int, float] = {}

    for sub_q in sub_queries:
        query_vec = processor.get_embeddings([sub_q])
        # _, indices = store.search(query_vec, k=top_k) 
        _, indices = store.hybrid_search(
            query_embedding=query_vec,
            query_text=sub_q,
            k=top_k,
        )
        for rank, doc_id in enumerate(indices[0].tolist()):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Sort by fused score descending, return top_k indices
    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return sorted_ids[:top_k]


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
            f"Desc: {c.get('description', '')[:180]}"
        )

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""
    You are ranking SHL assessments for retrieval evaluation.

    JOB REQUIREMENT:
    {query}

    CANDIDATES:
    {candidates_text}

    GUIDELINES:
    - Prefer assessments that directly match the role's key skills, level, and context.
    - It is OK to include personality/behaviour tests when the role involves people, teams, clients, or leadership.
    - It is OK to include cognitive/reasoning tests when the role involves analysis, data, problem solving, or communication.
    - Include technical skill tests for any specific tools or technologies mentioned.
    - If duration is specified, try to respect it, but do not exclude otherwise highly relevant assessments only for small duration differences.
    - When in doubt between several similar options, keep more rather than fewer (favor recall over strict filtering).

    Return ONLY a JSON array of {top_k_return} indices (0-based), ordered from most to least relevant.
    Example: [2, 0, 5, 1]
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
    seen = set()

    for i in selected_indices:
        if i not in seen:
            final.append(candidates[i])
            seen.add(i)
        if len(final) >= top_k_return:
            break

    # Safety fallback
    for i in range(len(candidates)):
        if len(final) >= top_k_return:
            break
        if i not in seen:
            final.append(candidates[i])
            seen.add(i)

    return final[:top_k_return]

def get_url_text(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if "view" in parts:
        return parts[-1]
    return parts[-1]

def evaluate_on_labelled_queries(top_k_retrieve: int = 10,top_k_final: int = 10,
    labelled_path: str = TRAIN_QUERIES_PATH, use_llm_rerank: bool = True) -> Dict[str, float]:

    labelled = _load_labelled_queries(labelled_path)
    assessments = _load_assessments()
    processor, store = _build_vector_store(assessments)
    gemini_model = _init_gemini()

    dataset_slugs = {get_url_text(a.get("url", "")) for a in assessments}

    total_relevant = 0
    retrieved_relevant = 0
    retrieved_hits = 0

    final_relevant = 0
    final_hits = 0
    processed_queries = 0

    for example in tqdm(labelled, desc="Evaluating labelled queries"):
        query = example.get("query", "").strip()
        all_relevant_urls = example.get("relevant_urls", [])

        if not query or not all_relevant_urls:
            continue

        # Normalize ground truth
        relevant_slugs = {
            get_url_text(u)
            for u in all_relevant_urls
            if get_url_text(u) in dataset_slugs
        }

        if not relevant_slugs:
            continue

        processed_queries += 1
        total_relevant += len(relevant_slugs)

        # ---------------- RETRIEVAL ----------------
        retrieved_indices = _retrieve_with_expansion(
            query=query,
            processor=processor,
            store=store,
            gemini_model=gemini_model,
            top_k=top_k_retrieve,
        )

        retrieved_slugs = {
            get_url_text(assessments[i].get("url", ""))
            for i in retrieved_indices if i >= 0
        }

        overlap = retrieved_slugs.intersection(relevant_slugs)

        retrieved_hits += 1 if overlap else 0
        retrieved_relevant += len(overlap)

        # ---------------- RERANK ----------------
        candidates = [assessments[i] for i in retrieved_indices if i >= 0]

        if use_llm_rerank:
            reranked = _rerank_with_llm_eval(
                gemini_model=gemini_model,
                query=query,
                candidates=candidates,
                top_k_return=top_k_final,
            )
        else:
            # Retrieval-only baseline
            reranked = candidates[:top_k_final]

        final_slugs = {
            get_url_text(a.get("url", ""))
            for a in reranked
        }

        final_overlap = final_slugs.intersection(relevant_slugs)

        final_hits += 1 if final_overlap else 0
        final_relevant += len(final_overlap)

    if total_relevant == 0 or processed_queries == 0:
        return {}

    return {
        f"retrieval_recall@{top_k_retrieve}":retrieved_relevant / total_relevant,
        f"retrieval_hit_rate@{top_k_retrieve}":retrieved_hits / processed_queries,
        f"recommendation_recall@{top_k_final}":final_relevant / total_relevant,
        f"recommendation_hit_rate@{top_k_final}":final_hits / processed_queries,
    }


if __name__ == "__main__":
    try:
        print("\n=== Labelled train-data evaluation ===") 
        metrics = evaluate_on_labelled_queries(use_llm_rerank=True)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}") 

    except Exception as e:
        print(f"Error evaluating labelled queries: {e}")