import json
import os
import uvicorn
import requests
import google.generativeai as genai
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List 
from playwright.sync_api import sync_playwright

from embedding import TextProcessor, VectorStore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, "scrapper", "output", "shl_individual_tests_enrich.json")
TOP_K_RETRIEVE = 50
TOP_K_RERANK   = 15
TOP_K_RETURN   = 10
MAX_QUERY_LEN  = 3000

processor   = None
assessments = None
gemini      = None
store       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, assessments, gemini, store

    print("Loading dataset...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print("Building embeddings...")
    processor = TextProcessor()
    texts = [processor.build_assessment_text(a) for a in assessments]
    embeddings = processor.get_embeddings(texts)

    print("Building FAISS + BM25 index...")
    store = VectorStore()
    store.create_index(embeddings, raw_texts=texts)

    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-2.5-flash")
    print("=== Ready ===")
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str


class Assessment(BaseModel):
    url:              str
    name:             str
    adaptive_support: str
    description:      str
    duration:         int
    remote_support:   str
    test_type:        List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]


def fetch_url_text(url: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0"})
            page.goto(url, wait_until="networkidle", timeout=30000)
            text = " ".join(page.inner_text("body").split())[:MAX_QUERY_LEN]
            browser.close()
            if len(text) < 100:
                raise HTTPException(status_code=400, detail="Could not extract enough content. Please paste the job description directly.")
            return text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")


def retrieve(query: str, k: int) -> List[dict]:
    query_vec = processor.get_embeddings([query])
    _, indices = store.hybrid_search(query_embedding=query_vec, query_text=query, k=k)
    return [assessments[i] for i in indices.flatten().tolist() if 0 <= i < len(assessments)]


def rerank(query: str, candidates: List[dict], top_k_return: int) -> List[dict]:
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
        response = gemini.generate_content(prompt)
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

    return final[:top_k_return]


def format_assessment(item: dict) -> Assessment:
    return Assessment(
        url=item.get("url", ""),
        name=item.get("name", ""),
        adaptive_support=item.get("adaptive_support", "No"),
        description=item.get("description", "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip(),
        duration=int(item.get("duration") or 0),
        remote_support=item.get("remote_testing", "No"),
        test_type=item.get("test_type", []),
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    query = request.query.strip()[:MAX_QUERY_LEN]
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if query.startswith("http://") or query.startswith("https://"):
        query = fetch_url_text(query)

    candidates = retrieve(query, k=TOP_K_RETRIEVE)
    if not candidates:
        raise HTTPException(status_code=404, detail="No assessments found.")

    final = rerank(query, candidates, top_k_return=TOP_K_RERANK)[:TOP_K_RETURN]

    return RecommendResponse(
        recommended_assessments=[format_assessment(a) for a in final]
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)