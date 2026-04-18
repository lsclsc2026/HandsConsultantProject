from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

from app.core.config import settings
from app.rag.graph_retriever import GraphRetriever
from app.rag.knowledge_loader import KnowledgeLoader
from app.rag.types import RetrievedChunk


class HybridRetriever:
    def __init__(self) -> None:
        self.cfg = settings.rag
        self.loader = KnowledgeLoader()
        self.graph = GraphRetriever()

        self.docs: list[tuple[str, str]] = []
        for txt in self.loader.load_rules():
            self.docs.append(("rule", txt))
        for txt in self.loader.load_qa():
            self.docs.append(("qa", txt))

        self.texts = [item[1] for item in self.docs]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.doc_matrix = None
        self.bm25 = None

        if self.texts:
            self.doc_matrix = self.vectorizer.fit_transform(self.texts)
            tokenized = [list(jieba.cut(text)) for text in self.texts]
            self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v = float(scores.min())
        max_v = float(scores.max())
        if abs(max_v - min_v) < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    def retrieve(self, query: str, profile_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.texts:
            return []

        top_k = top_k or self.cfg.hybrid_top_k
        expanded_query = f"{query} {profile_text}".strip()

        dense_scores = np.zeros(len(self.texts), dtype=np.float32)
        bm25_scores = np.zeros(len(self.texts), dtype=np.float32)

        if self.doc_matrix is not None:
            q_vec = self.vectorizer.transform([expanded_query])
            dense_scores = (self.doc_matrix @ q_vec.T).toarray().reshape(-1).astype(np.float32)

        if self.bm25 is not None:
            bm25_scores = np.array(self.bm25.get_scores(list(jieba.cut(expanded_query))), dtype=np.float32)

        dense_norm = self._normalize(dense_scores)
        bm25_norm = self._normalize(bm25_scores)

        combined = (
            self.cfg.dense_weight * dense_norm
            + self.cfg.bm25_weight * bm25_norm
        )

        top_indices = np.argsort(-combined)[:top_k]
        results: list[RetrievedChunk] = []
        for idx in top_indices:
            source, text = self.docs[int(idx)]
            results.append(
                RetrievedChunk(
                    text=text,
                    source=source,
                    score=float(combined[int(idx)]),
                )
            )

        graph_hits = self.graph.retrieve(query, profile_text, top_k=max(1, top_k // 3))
        for rank, hit in enumerate(graph_hits):
            score = self.cfg.graph_weight * (1.0 - rank * 0.08)
            results.append(RetrievedChunk(text=hit, source="graph", score=max(score, 0.05)))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
