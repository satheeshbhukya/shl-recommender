import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

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


_ROLE_CONTEXT_BY_NAME = {
    # Ability & Aptitude — Verify series
    "verbal ability":       "suitable for content writer copywriter analyst consultant manager communication roles",
    "verbal reasoning":     "suitable for content writer copywriter analyst consultant manager communication roles",
    "numerical ability":    "suitable for data analyst finance banking accounting administrative assistant clerk roles",
    "numerical reasoning":  "suitable for data analyst finance banking accounting administrative assistant clerk roles",
    "numerical calculation":"suitable for data analyst finance banking accounting administrative assistant clerk roles",
    "inductive reasoning":  "suitable for analyst manager consultant strategic thinking marketing data science roles",
    "deductive reasoning":  "suitable for analyst manager consultant logical thinking professional roles",
    "working with information": "suitable for administrative assistant clerical data entry analyst roles",
    "following instructions":   "suitable for administrative assistant clerical operations roles",
    "general ability":      "suitable for graduate entry level professional analyst manager roles",
    "verify g+":            "suitable for graduate entry level professional analyst manager roles",
    "verify interactive g+":"suitable for graduate entry level professional analyst manager roles",
    "process monitoring":   "suitable for operations analyst administrative quality assurance roles",

    # Personality & Behavior — OPQ / Leadership series
    "occupational personality": "suitable for leadership COO CEO director executive manager culture fit senior professional roles",
    "opq leadership":           "suitable for COO CEO director executive senior manager leadership roles",
    "opq team":                 "suitable for team manager COO senior leader collaboration culture roles",
    "enterprise leadership":    "suitable for COO CEO director VP executive senior manager leadership roles",
    "global skills":            "suitable for COO executive manager international global cross cultural leadership roles",
    "sales profiler":           "suitable for sales manager business development account manager roles",
    "sales transformation":     "suitable for sales manager business development territory manager roles",
    "sales interview":          "suitable for sales representative business development account manager roles",
    "opq mq sales":             "suitable for sales manager territory manager business development roles",

    # Communication / Language
    "interpersonal communications": "suitable for sales customer service team collaboration business analyst software developer roles requiring communication",
    "business communication":       "suitable for sales customer service administrative professional communication roles",
    "english comprehension":        "suitable for content writer sales customer service administrative communication roles",
    "written english":              "suitable for content writer copywriter editor documentation technical writer roles",
    "reading comprehension":        "suitable for content writer analyst consultant professional roles",
    "svar":                         "suitable for sales customer service call centre communication spoken english roles",
    "spoken english":               "suitable for sales customer service call centre communication roles",

    # Data / Analytics
    "tableau":              "suitable for data analyst business intelligence BI reporting analyst roles",
    "data warehousing":     "suitable for data analyst data engineer ETL SQL database architect roles",
    "microsoft excel":      "suitable for data analyst financial analyst administrative assistant marketing manager roles",
    "sql server analysis":  "suitable for data analyst business intelligence SSAS OLAP reporting roles",

    # Marketing
    "marketing":            "suitable for marketing manager digital marketing brand manager campaign manager roles",
    "digital advertising":  "suitable for marketing manager digital marketing performance marketing roles",
    "search engine optimization": "suitable for SEO specialist content writer digital marketing manager roles",
    "writex":               "suitable for sales marketing content writer email communication roles",
}


_NOISE_RE = re.compile(
    r"Your use of this assessment.*?shl\.com/legal/[^\r\n]*"
    r"|Report Language Availability:.*"
    r"|Read more on https?://\S+",
    re.IGNORECASE | re.DOTALL,
)


def _clean_description(text: str) -> str:
    return _NOISE_RE.sub("", text).strip()


def _slug_to_keywords(url: str) -> str:
    """'core-java-entry-level-new' → 'core java entry level'"""
    slug = url.rstrip("/").split("/")[-1]
    words = [w for w in slug.replace("-", " ").split()
             if w not in ("new", "v1", "v2")]
    return " ".join(words)


def _get_role_context(item: dict) -> str:
    """Look up role context string for this assessment based on name patterns."""
    name_lower = item.get("name", "").lower()
    for pattern, context in _ROLE_CONTEXT_BY_NAME.items():
        if pattern in name_lower:
            return context
    return ""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class TextProcessor:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def build_assessment_text(self, item: dict) -> str:
        name        = item.get("name", "")
        description = _clean_description(item.get("description", ""))
        test_types  = item.get("test_type", [])
        duration    = item.get("duration", 0)
        url         = item.get("url", "")

        slug_keywords = _slug_to_keywords(url)

        # Vocab bridge — only for types with genuine vocabulary gap
        type_vocab = " ".join(
            TEST_TYPE_VOCAB[t] for t in test_types if t in TEST_TYPE_VOCAB
        ).strip()

        # Role context — bridges job title in query to measurement in doc
        role_context = _get_role_context(item)

        duration_str = str(duration) if duration and duration > 0 else ""

        parts = [
            f"Assessment Name: {name}",
            f"Keywords: {slug_keywords}",
            f"Description: {description}",
            f"Test Type: {', '.join(test_types)}",
            f"Remote Testing: {item.get('remote_testing', '')}",
            f"Adaptive Support: {item.get('adaptive_support', '')}",
        ]
        if duration_str:
            parts.append(f"Duration: {duration_str} minutes")
        if type_vocab:
            parts.append(f"Relevant For: {type_vocab}")
        if role_context:
            parts.append(f"Suitable For: {role_context}")

        return " | ".join(p for p in parts if p)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.dimension: Optional[int] = None
        self._bm25: Optional[BM25Okapi] = None

    def create_index(self, embeddings: np.ndarray, raw_texts: Optional[List[str]] = None):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        if raw_texts is not None:
            self._bm25 = BM25Okapi([_tokenize(t) for t in raw_texts])
        return self.index

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Dense-only — kept for self-retrieval eval."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        k = min(k, self.index.ntotal)
        return self.index.search(query_embedding, k)

    def reset(self):
        self.index = None
        self.dimension = None
        self._bm25 = None