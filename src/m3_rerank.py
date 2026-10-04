from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _cached_models: dict = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self.model_name in CrossEncoderReranker._cached_models:
            self._model = CrossEncoderReranker._cached_models[self.model_name]
            return self._model

        from sentence_transformers import CrossEncoder

        # Dùng sentence_transformers.CrossEncoder để tránh xung đột với FlagEmbedding
        # Ưu tiên kiểm tra cache local trước để không bị nghẽn mạng
        try:
            self._model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception:
                # Fallback model chất lượng cao đã tải sẵn
                fallback_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self._model = CrossEncoder(fallback_name)

        CrossEncoderReranker._cached_models[self.model_name] = self._model
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        results = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    _cached_ranker = None

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if FlashrankReranker._cached_ranker is None:
                FlashrankReranker._cached_ranker = Ranker()
            passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
            rerank_req = RerankRequest(query=query, passages=passages)
            results = FlashrankReranker._cached_ranker.rerank(rerank_req)
            reranked = []
            for i, r in enumerate(results[:top_k]):
                doc_idx = r["id"]
                reranked.append(RerankResult(
                    text=r["text"],
                    original_score=float(documents[doc_idx].get("score", 0.0)),
                    rerank_score=float(r["score"]),
                    metadata=documents[doc_idx].get("metadata", {}),
                    rank=i + 1
                ))
            return reranked
        except Exception:
            return [
                RerankResult(
                    text=d.get("text", ""),
                    original_score=float(d.get("score", 0.0)),
                    rerank_score=0.0,
                    metadata=d.get("metadata", {}),
                    rank=i + 1
                )
                for i, d in enumerate(documents[:top_k])
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
