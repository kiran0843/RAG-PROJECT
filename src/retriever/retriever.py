"""Retriever: metric metadata filter or hybrid vector search."""

import logging
import re
from typing import Any

from src.embeddings.encoder import EmbeddingEncoder
from src.retriever.query_expansion import expand_for_narrative
from src.retriever.query_intent import QueryIntent, classify_query_intent
from src.vector_store.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


_FINANCIAL_TOKENS = frozenset([
    "net", "profit", "loss", "income", "earnings", "tax", "comprehensive",
])
_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    """Extract word tokens from text."""
    return set(_TOKEN_RE.findall(text.lower()))


def _direct_metric_retrieval(
    store: FAISSStore,
    intent: QueryIntent,
) -> list[dict[str, Any]]:
    """Filter structured chunks by metadata. Returns matching chunks only."""
    def predicate(meta: dict[str, Any]) -> bool:
        if not meta.get("is_structured_metric"):
            return False
        if meta.get("metric_name") != intent.target_metric:
            return False
        if intent.year is not None and meta.get("year") != intent.year:
            return False
        if intent.report_type is not None and meta.get("report_type") != intent.report_type:
            return False
        return True

    return store.filter_by_metadata(predicate)


def _apply_hybrid_scoring(
    query: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply token-based hybrid scoring for vector search fallback."""
    query_tokens = _tokenize(query)
    query_financial = query_tokens & _FINANCIAL_TOKENS

    scored: list[dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        text = meta.get("text") or ""
        chunk_tokens = _tokenize(text)
        chunk_financial = chunk_tokens & _FINANCIAL_TOKENS
        overlap_count = len(query_financial & chunk_financial)
        keyword_score = min(1.0, overlap_count / 3.0)

        sim = max(0.0, float(r.get("score", 0.0)))
        final = 0.6 * sim + 0.4 * keyword_score

        r_copy = dict(r)
        r_copy["similarity_score"] = sim
        r_copy["keyword_score"] = keyword_score
        r_copy["final_score"] = final
        r_copy["score"] = final
        scored.append(r_copy)
    return scored


class Retriever:
    """Retrieves chunks: metric metadata filter or hybrid vector search."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        store: FAISSStore,
        *,
        top_k: int = 8,
        similarity_threshold: float = 0.35,
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve chunks for the query.

        If query targets a known financial metric: returns only matching
        structured chunks (no vector search).
        Otherwise: hybrid vector search.
        """
        k = top_k if top_k is not None else self._top_k
        threshold = similarity_threshold if similarity_threshold is not None else self._similarity_threshold

        if not query or not query.strip():
            return []

        intent = classify_query_intent(query)

        # 1. Direct metric retrieval when intent targets a known metric.
        if intent and intent.target_metric:
            direct = _direct_metric_retrieval(self._store, intent)
            if direct:
                for r in direct:
                    r.setdefault("similarity_score", 1.0)
                    r.setdefault("keyword_score", 1.0)
                    r.setdefault("final_score", 1.0)
                logger.debug(
                    "direct metric: metric=%s year=%s -> %d chunks",
                    intent.target_metric,
                    intent.year,
                    len(direct),
                )
                return direct

        search_query = expand_for_narrative(query)
        query_embedding = self._encoder.embed_query(search_query)
        results = self._store.search(query_embedding, top_k=k)
        if not results:
            return []

        scored = _apply_hybrid_scoring(search_query, results)
        if not scored:
            return []

        scored.sort(key=lambda r: float(r.get("final_score", 0.0)), reverse=True)
        best_final = float(scored[0].get("final_score", 0.0))
        if best_final < threshold:
            return []

        filtered = [
            r for r in scored if float(r.get("final_score", 0.0)) >= threshold
        ][:k]

        logger.debug(
            "hybrid search: %d results above threshold",
            len(filtered),
        )
        return filtered
