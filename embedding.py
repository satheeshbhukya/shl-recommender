import json
import pickle
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

INPUT_PATH    = "../scrapper/output/shl_individual_tests.json"
INDEX_PATH    = "../output/faiss_index.bin"
METADATA_PATH = "../output/metadata.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"


class TextProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def build_assessment_text(self, item: dict) -> str:
        test_types = ", ".join(item.get("test_type", []))
        parts = [
            f"Assessment Name: {item.get('name', '')}",
            f"Description: {item.get('description', '')}",
            f"Test Type: {test_types}",
            f"Remote Testing: {item.get('remote_testing', '')}",
            f"Adaptive Support: {item.get('adaptive_support', '')}",
            f"Duration: {item.get('duration', '')}",
        ]
        return " | ".join(p for p in parts if p)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    def __init__(self):
        self.index     = None
        self.dimension = None

    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return self.index

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices

    def save_index(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)

    def load_index(self, filepath: str):
        self.index     = faiss.read_index(filepath)
        self.dimension = self.index.d

    def reset(self):
        self.index     = None
        self.dimension = None


def main():
 
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)
    processor = TextProcessor()
    texts     = [processor.build_assessment_text(a) for a in assessments]
    embeddings = processor.get_embeddings(texts)
    store = VectorStore()
    store.create_index(embeddings)
    store.save_index(INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(assessments, f)

if __name__ == "__main__":
    main()