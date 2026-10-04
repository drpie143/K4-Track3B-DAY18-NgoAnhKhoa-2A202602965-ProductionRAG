from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        # underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
        # BM25 tokenize bằng split(" ") → replace("_", " ") để BM25 và truy vấn khớp chuẩn.
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            tokens = segment_vietnamese(chunk.get("text", "")).lower().split()
            self.corpus_tokens.append(tokens)

        from rank_bm25 import BM25Okapi
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_indices = [i for i, s in enumerate(scores) if s > 0]
        if not scored_indices:
            scored_indices = list(range(len(scores)))

        scored_indices.sort(key=lambda i: scores[i], reverse=True)
        results = []
        for i in scored_indices[:top_k]:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    _cached_encoder = None

    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if DenseSearch._cached_encoder is not None:
            self._encoder = DenseSearch._cached_encoder
            return self._encoder

        from sentence_transformers import SentenceTransformer

        try:
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            # Fallback sang model đã được cache sẵn trong môi trường local
            fallback = "all-MiniLM-L6-v2"
            self._encoder = SentenceTransformer(fallback)

        DenseSearch._cached_encoder = self._encoder
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        texts = [c.get("text", "") for c in chunks] if chunks else []
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=False) if texts else []
        dim = int(vectors[0].shape[0]) if len(vectors) > 0 else EMBEDDING_DIM

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        if not chunks or len(vectors) == 0:
            return

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")}
            ))

        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        query_vector = encoder.encode(query).tolist()
        try:
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            points = response.points
        except Exception:
            try:
                points = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
            except Exception:
                return []

        results = []
        for pt in points:
            payload = pt.payload or {}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=float(pt.score) if hasattr(pt, "score") else 0.0,
                metadata=payload,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    rrf_scores = {}  # text -> {"score": float, "result": SearchResult}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    fused_results = []
    for item in sorted_items[:top_k]:
        res = item["result"]
        fused_results.append(SearchResult(
            text=res.text,
            score=round(item["score"], 6),
            metadata=res.metadata,
            method="hybrid"
        ))
    return fused_results


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
