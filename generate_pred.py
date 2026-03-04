import json
import os
import pandas as pd
from typing import List
from embedding import TextProcessor, VectorStore

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "scrapper", "output", "shl_individual_tests_enrich.json")
TEST_PATH    = os.path.join(BASE_DIR, "dataset", "Test-Set.xlsx")
OUTPUT_PATH  = os.path.join(BASE_DIR, "dataset", "predictions.csv")

TOP_K_RETRIEVE = 50
TOP_K_RERANK   = 15
TOP_K_RETURN   = 10

try:
    from google import genai
    def _init_gemini():
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    def _generate(client, prompt):
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
except ImportError:
    import google.generativeai as genai
    def _init_gemini():
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai.GenerativeModel("gemini-2.5-flash")
    def _generate(client, prompt):
        return client.generate_content(prompt).text.strip()


def _load_assessments() -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_vector_store(assessments):
    processor = TextProcessor()
    texts = [processor.build_assessment_text(a) for a in assessments]
    embeddings = processor.get_embeddings(texts)
    store = VectorStore()
    store.create_index(embeddings, raw_texts=texts)
    return processor, store


def retrieve(processor, store, assessments, query: str, k: int) -> List[dict]:
    query_vec = processor.get_embeddings([query])
    _, indices = store.hybrid_search(query_embedding=query_vec, query_text=query, k=k)
    return [assessments[i] for i in indices.flatten().tolist() if 0 <= i < len(assessments)]


def rerank(client, query: str, candidates: List[dict], top_k_return: int) -> List[dict]:
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
        raw = _generate(client, prompt).replace("```json", "").replace("```", "").strip()
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

    return final[:top_k_return]


def generate_predictions():
    print("Loading assessments...")
    assessments = _load_assessments()

    print("Building vector store...")
    processor, store = _build_vector_store(assessments)

    print("Initializing Gemini...")
    client = _init_gemini()

    df = pd.read_excel(TEST_PATH)
    queries = df["Query"].tolist()

    rows = []
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}/{len(queries)}: {query[:80]}...")
        candidates = retrieve(processor, store, assessments, query, k=TOP_K_RETRIEVE)
        final      = rerank(client, query, candidates, top_k_return=TOP_K_RERANK)[:TOP_K_RETURN]

        for a in final:
            rows.append({"Query": query, "Assessment_url": a.get("url", "")})

        print(f"  → {len(final)} recommendations")

    out = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print(f"Total rows: {len(out)}")


if __name__ == "__main__":
    generate_predictions()