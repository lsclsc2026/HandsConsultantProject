from __future__ import annotations

import os
import logging
from pathlib import Path

import jieba

from app.core.config import settings
from app.rag.types import RetrievedChunk


try:
    from sentence_transformers import CrossEncoder
except Exception:  # noqa: BLE001
    CrossEncoder = None


logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        self.top_k = settings.rag.rerank_top_k
        self.model = None
        self.model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.local_only = os.getenv("RERANKER_LOCAL_ONLY", "1") == "1"
        self.enabled = os.getenv("RERANKER_ENABLED", "1") == "1"
        self._load_attempted = False

    def _ensure_model_loaded(self) -> None:
        if self._load_attempted or not self.enabled:
            return
        self._load_attempted = True

        if CrossEncoder is None:
            logger.info("sentence-transformers 不可用，使用词法重排降级。")
            return

        if self.local_only and not Path(self.model_name).exists():
            logger.info(
                "本地未找到重排模型，当前使用词法重排。可设置 RERANKER_LOCAL_ONLY=0 联网下载。model=%s",
                self.model_name,
            )
            return

        try:
            self.model = CrossEncoder(
                self.model_name,
                local_files_only=self.local_only,
            )
            logger.info("CrossEncoder 重排模型加载成功: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            self.model = None
            logger.warning(
                "CrossEncoder 加载失败，回退词法重排。model=%s local_only=%s err=%s",
                self.model_name,
                self.local_only,
                exc,
            )

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q_tokens = set(jieba.cut(query))
        t_tokens = set(jieba.cut(text))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & t_tokens)
        return overlap / len(q_tokens)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        top_k = top_k or self.top_k

        self._ensure_model_loaded()

        if self.model is not None:
            pairs = [[query, c.text] for c in chunks]
            scores = self.model.predict(pairs)
            rescored = [
                RetrievedChunk(text=c.text, source=c.source, score=float(s))
                for c, s in zip(chunks, scores, strict=True)
            ]
            rescored.sort(key=lambda x: x.score, reverse=True)
            return rescored[:top_k]

        rescored = [
            RetrievedChunk(text=c.text, source=c.source, score=self._lexical_score(query, c.text))
            for c in chunks
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
