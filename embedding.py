import re
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-mpnet-base-v2"
RRF_K = 60

TEST_TYPE_VOCAB = {
    "Personality & Behavior": (
        "personality behaviour culture fit teamwork collaboration leadership "
        "interpersonal style work style values motivation"
    ),
    "Ability & Aptitude": (
        "reasoning analytical thinking problem solving cognitive ability "
        "numerical verbal logical inductive deductive aptitude"
    ),
    "Biodata & Situational Judgement": (
        "situational judgement scenario decision making background experience"
    ),
    "Development & 360": (
        "360 feedback development multi-rater leadership growth"
    ),
}

_NOISE_RE_1 = re.compile(
    r"Your use of this assessment.*?shl\.com/legal/[^\r\n]*",
    re.IGNORECASE | re.DOTALL,
)
_NOISE_RE_2 = re.compile(
    r"Report Language Availability:.*|Read more on https?://\S+",
    re.IGNORECASE,
)


def extract_url_slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _clean_description(text: str) -> str:
    text = _NOISE_RE_1.sub("", text)
    text = _NOISE_RE_2.sub("", text)
    return text.strip()


def _slug_to_keywords(url: str) -> str:
    slug = extract_url_slug(url)
    words = [w for w in slug.replace("-", " ").split() if w not in ("new", "v1", "v2")]
    return " ".join(words)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


class TextProcessor:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def _build_metadata_text(self, item: dict) -> str:
        name         = item.get("name", "")
        description  = _clean_description(item.get("description", ""))
        test_types   = item.get("test_type", [])
        duration     = item.get("duration", 0)
        url          = item.get("url", "")
        role_summary = item.get("role_summary", "").strip()

        slug_keywords = _slug_to_keywords(url)
        type_vocab    = " ".join(TEST_TYPE_VOCAB[t] for t in test_types if t in TEST_TYPE_VOCAB).strip()
        duration_str  = str(duration) if duration and int(duration) > 0 else ""

        parts = [
            f"Assessment Name: {name}",
            f"Keywords: {slug_keywords}",
            f"Description: {description}",
            f"Test Type: {', '.join(test_types)}",
            f"Remote Testing: {item.get('remote_testing', '')}",
            f"Adaptive Support: {item.get('adaptive_support', '')}",
            f"Role Context: {role_summary}",
        ]
        if duration_str:
            parts.append(f"Duration: {duration_str} minutes")
        if type_vocab:
            parts.append(f"Relevant For: {type_vocab}")

        return " | ".join(p for p in parts if p)

    def build_assessment_text(self, item: dict) -> str:
        return self._build_metadata_text(item)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.dimension: Optional[int] = None
        self._bm25: Optional[BM25Okapi] = None

    def create_index(self, embeddings: np.ndarray, raw_texts: Optional[List[str]] = None) -> faiss.Index:
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        if raw_texts is not None:
            self._bm25 = BM25Okapi([_tokenize(t) for t in raw_texts])
        return self.index

    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        k_retrieve = min(k * 8, self.index.ntotal)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        dense_scores, dense_indices = self.index.search(query_embedding, k_retrieve)
        dense_indices = dense_indices[0]

        bm25_scores_full = self._bm25.get_scores(_tokenize(query_text)) if self._bm25 is not None else np.zeros(self.index.ntotal)
        bm25_ranked      = np.argsort(bm25_scores_full)[::-1]
        bm25_top         = bm25_ranked[:k_retrieve]

        candidate_ids  = sorted(set(dense_indices.tolist()).union(set(bm25_top.tolist())))
        dense_rank_map = {int(idx): rank for rank, idx in enumerate(dense_indices)}
        bm25_rank_map  = {int(idx): rank for rank, idx in enumerate(bm25_ranked[:k_retrieve])}

        rrf_scores = np.array([
            _rrf_score(dense_rank_map.get(idx, k_retrieve)) + _rrf_score(bm25_rank_map.get(idx, k_retrieve))
            for idx in candidate_ids
        ])

        top_k_idx  = np.argsort(rrf_scores)[::-1][:k]
        top_ids    = np.array([candidate_ids[i] for i in top_k_idx], dtype=np.int64)
        top_scores = rrf_scores[top_k_idx].astype(np.float32)

        return (top_scores.reshape(1, -1), top_ids.reshape(1, -1))

    def reset(self):
        self.index     = None
        self.dimension = None
        self._bm25     = None