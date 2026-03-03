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



# Path to the labelled train set provided in the repo
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
    return genai.GenerativeModel("gemini-2.0-flash")


def _hit_rate_at_k(ranks: List[int], k: int) -> float:
    hits = sum(1 for r in ranks if 0 <= r < k)
    return hits / len(ranks) if ranks else 0.0


def _mrr_at_k(ranks: List[int], k: int) -> float:
    total = 0.0
    for r in ranks:
        if 0 <= r < k:
            total += 1.0 / (r + 1)
    return total / len(ranks) if ranks else 0.0


def evaluate_self_retrieval(top_ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    assessments = _load_assessments()
    processor, store = _build_vector_store(assessments)

    self_ranks: List[int] = []

    for idx, item in enumerate(tqdm(assessments, desc="Evaluating self-retrieval")):
        query_text = processor.build_assessment_text(item)
        query_vec = processor.get_embeddings([query_text])
        scores, indices = store.search(query_vec, k=max(top_ks))

        retrieved_indices = list(indices[0])
        try:
            rank = retrieved_indices.index(idx)
        except ValueError:
            rank = -1
        self_ranks.append(rank)

    metrics: Dict[str, float] = {}
    for k in top_ks:
        metrics[f"hit_rate@{k}"] = _hit_rate_at_k(self_ranks, k)
        metrics[f"mrr@{k}"] = _mrr_at_k(self_ranks, k)
    return metrics


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

def _retrieve_with_expansion(
    query: str,
    processor: TextProcessor,
    store: VectorStore,
    gemini_model,
    top_k: int,
    rrf_k: int = 60,
) -> List[int]: 

    """Expand query into sub-queries, retrieve for each, fuse with RRF.

    For short queries this is identical to a plain search — expand_query
    returns [query] unchanged when word count < threshold.
    """
    sub_queries = expand_query(query, gemini_model)

    rrf_scores: Dict[int, float] = {}

    for sub_q in sub_queries:
        query_vec = processor.get_embeddings([sub_q])
        _, indices = store.search(query_vec, k=top_k)
        for rank, doc_id in enumerate(indices[0].tolist()):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Sort by fused score descending, return top_k indices
    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return sorted_ids[:top_k]



def _rerank_with_llm_eval(
    gemini_model,
    query: str,
    candidates: List[dict],
    top_k_return: int,
) -> List[dict]:

    candidate_lines = []
    for i, c in enumerate(candidates):
        test_types = ", ".join(c.get("test_type", []))
        candidate_lines.append(
            f"{i}. Name: {c.get('name', '')} | TestType: {test_types} | "
            f"Remote: {c.get('remote_testing')} | Adaptive: {c.get('adaptive_support')} | "
            f"Desc: {c.get('description', '')[:150]}"
        )

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""ROLE: You are a strict HR assessment selector at SHL.

JOB REQUIREMENT: {query}

CANDIDATE ASSESSMENTS TO EVALUATE:
{candidates_text}

CONSTRAINTS:
1. MUST include assessments for every specific technology/skill mentioned in the query
2. MUST NOT include personality tests unless collaboration/teamwork/leadership is mentioned
3. MUST NOT include cognitive tests unless analytical/reasoning/problem-solving is mentioned
4. SELECT minimum 5, maximum 10 assessments
5. ORDER by relevance — exact skill matches first

RESPOND with only a JSON array of assessment indices (0-based).
Example output: [1, 4, 0, 7, 3]

Return only the JSON array, nothing else."""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        selected_indices = json.loads(raw)
        selected_indices = [i for i in selected_indices if 0 <= i < len(candidates)]
        return [candidates[i] for i in selected_indices[:top_k_return]]
    except Exception:
        return candidates[:top_k_return]

def get_url_text(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if "view" in parts:
        return parts[-1]
    return parts[-1]

def evaluate_on_labelled_queries(
    top_k_retrieve: int = 20,
    top_k_final: int = 10,
    labelled_path: str = TRAIN_QUERIES_PATH
) -> Dict[str, float]:

    labelled = _load_labelled_queries(labelled_path)

    assessments = _load_assessments()
    processor, store = _build_vector_store(assessments)
    gemini_model = _init_gemini()

    # FIX: build slug set of what actually exists in your index
    # so solution bundles not in catalog don't penalise recall
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

        # FIX: filter relevant_urls to only assessments present in your index
        relevant_urls = [
            u for u in all_relevant_urls
            if get_url_text(u) in dataset_slugs
        ]

        if not relevant_urls:
            continue

        processed_queries+=1
        # Use unique relevant targets to match set-based overlap logic.
        relevant_texts = set(get_url_text(u) for u in relevant_urls)
        total_relevant += len(relevant_texts) 

        retrieved_indices = _retrieve_with_expansion(
            query=query,
            processor=processor,
            store=store,
            gemini_model=gemini_model,
            top_k=top_k_retrieve,
        )
        retrieved_urls = [assessments[i].get("url", "") for i in retrieved_indices if i >= 0]


        # Retrieval stage metrics  
        retrieved_text = set(get_url_text(u) for u in retrieved_urls)

        overlap = set(retrieved_text).intersection(set(relevant_texts)) 
        retrieved_hits += 1 if overlap else 0 
        retrieved_relevant += len(overlap) if overlap else 0

        # Build candidate list and run LLM reranker to mirror production behaviour.
        candidates = [assessments[i] for i in retrieved_indices if i >= 0]
        # reranked = _rerank_with_llm_eval(
        #     gemini_model=gemini_model,
        #     query=query,
        #     candidates=candidates,
        #     top_k_return=top_k_final,
        # )
        # final_urls = [a.get("url", "") for a in reranked]

        # final_overlap = set(final_urls).intersection(set(relevant_urls))
        # final_hits += 1 if final_overlap else 0
        # final_relevant += len(final_overlap) if final_overlap else 0

    if total_relevant == 0:
        return {}

    num_queries = processed_queries

    retrieval_recall = retrieved_relevant / total_relevant
    retrieval_hit_rate = retrieved_hits / num_queries if num_queries else 0.0

    # final_recall = final_relevant / total_relevant
    # final_hit_rate = final_hits / num_queries if num_queries else 0.0

    return {
        "retrieval_recall@{}".format(top_k_retrieve): retrieval_recall,
        "retrieval_hit_rate@{}".format(top_k_retrieve): retrieval_hit_rate,
        # "recommendation_recall@{}".format(top_k_final): final_recall,
        # "recommendation_hit_rate@{}".format(top_k_final): final_hit_rate,
    }


if __name__ == "__main__":
    # print("=== Self-retrieval evaluation ===")
    # self_metrics = evaluate_self_retrieval()
    # for k, v in self_metrics.items():
    #     print(f"{k}: {v:.4f}")

    try:
        print("\n=== Labelled train-data evaluation ===") 
        metrics = evaluate_on_labelled_queries()
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}") 

    except Exception as e:
        print(f"Error evaluating labelled queries: {e}")