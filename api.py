import json
import pickle
import os
import numpy as np
import faiss 
import uvicorn
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import List 
from contextlib import asynccontextmanager

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH     = os.path.join(BASE_DIR, "output", "faiss_index.bin")
METADATA_PATH  = os.path.join(BASE_DIR, "output", "metadata.pkl")
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K_RETRIEVE = 30  
TOP_K_RETURN   = 10

embedder = None
index = None
assessments = None
gemini = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, index, assessments, gemini
    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        assessments = pickle.load(f)
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-2.5-flash")
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
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())[:4000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")


def embed_query(text: str) -> np.ndarray:
    vec = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def faiss_search(query_vec: np.ndarray, k: int) -> list:
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = assessments[idx].copy()
        item["_score"] = float(score)
        results.append(item)
    return results


def rerank_with_llm(query: str, candidates: list) -> list:
    candidate_lines = []
    for i, c in enumerate(candidates):
        test_types = ", ".join(c.get("test_type", []))
        candidate_lines.append(
            f"{i}. Name: {c['name']} | TestType: {test_types} | "
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

    response = gemini.generate_content(prompt)
    raw = response.text.strip()
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        selected_indices = json.loads(raw)
        selected_indices = [i for i in selected_indices if 0 <= i < len(candidates)]
        return [candidates[i] for i in selected_indices[:TOP_K_RETURN]]
    except Exception:
        return candidates[:TOP_K_RETURN]


def format_assessment(item: dict) -> Assessment:
    return Assessment(
        url=item.get("url", ""),
        name=item.get("name", ""),
        adaptive_support=item.get("adaptive_support", "No"),
        description=item.get("description", ""),
        duration=int(item.get("duration") or 0),
        remote_support=item.get("remote_testing", "No"),
        test_type=item.get("test_type", []),
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if query.startswith("http://") or query.startswith("https://"):
        query = fetch_url_text(query)
    query_vec = embed_query(query)
    candidates = faiss_search(query_vec, k=TOP_K_RETRIEVE)
    if not candidates:
        raise HTTPException(status_code=404, detail="No assessments found.")
    reranked = rerank_with_llm(query, candidates)
    return RecommendResponse(
        recommended_assessments=[format_assessment(a) for a in reranked]
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000,reload=True)