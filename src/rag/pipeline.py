"""RAG pipeline: retrieve, build prompt, generate via LLM."""

import re
from typing import Any

from src.embeddings.encoder import EmbeddingEncoder
from src.llm.ollama_client import OllamaClient
from src.prompts.rag_prompt import RAG_SYSTEM_INSTRUCTIONS, build_rag_user_message
from src.retriever.retriever import Retriever
from src.vector_store.faiss_store import FAISSStore

NO_CONTEXT_MESSAGE = "I cannot find this in the provided documents."
_VALUE_RE = re.compile(r"\b(?:was|were)\s+(.+?)\s*\.\s*$", re.IGNORECASE | re.DOTALL)


def _format_metric_dual_response(results: list[dict[str, Any]]) -> str | None:
    """Format standalone+consolidated metric values. Returns formatted string or None."""
    if not results:
        return None
    metas = [r.get("metadata", {}) or {} for r in results]
    if not all(m.get("is_structured_metric") for m in metas):
        return None
    metric_names = {m.get("metric_name") for m in metas}
    if len(metric_names) != 1:
        return None
    report_types = {m.get("report_type") for m in metas}
    if "standalone" not in report_types or "consolidated" not in report_types:
        return None

    metric_name = next(iter(metric_names))
    by_year: dict[int, dict[str, str]] = {}
    for r in results:
        meta = r.get("metadata", {}) or {}
        rt = meta.get("report_type")
        year = meta.get("year")
        text = meta.get("text", "")
        if not rt or year is None:
            continue
        m = _VALUE_RE.search(text.strip())
        val = m.group(1).strip() if m else ""
        if year not in by_year:
            by_year[year] = {}
        by_year[year][rt] = val

    lines: list[str] = []
    for year in sorted(by_year.keys(), reverse=True):
        d = by_year[year]
        if "standalone" in d and "consolidated" in d:
            lines.append(f"For FY {year}:")
            lines.append(f"- Standalone {metric_name}: {d['standalone']}.")
            lines.append(f"- Consolidated {metric_name}: {d['consolidated']}.")

    return "\n".join(lines) if lines else None


def _format_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source metadata for each retrieved chunk."""
    return [
        {
            "text": r.get("metadata", {}).get("text", ""),
            "page_number": r.get("metadata", {}).get("page_number"),
            "chunk_id": r.get("metadata", {}).get("chunk_id"),
            "source_path": r.get("metadata", {}).get("source_path"),
        }
        for r in results
    ]


def _format_similarity_scores(results: list[dict[str, Any]]) -> list[float]:
    """Extract similarity scores from search results."""
    return [float(r.get("score", 0.0)) for r in results]


class RAGPipeline:
    """
    RAG pipeline: accept query → retrieve top chunks → build prompt → call Ollama
    → return { answer, sources, similarity_scores }.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        store: FAISSStore,
        llm: OllamaClient,
        *,
        top_k: int = 5,
        context_max_chars: int = 3000,
        system_prompt: str | None = None,
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._context_max_chars = context_max_chars
        self._system_prompt = system_prompt or RAG_SYSTEM_INSTRUCTIONS
        self._retriever = Retriever(encoder=encoder, store=store, top_k=top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return top-k chunks with score and metadata."""
        if not (query or "").strip():
            return []
        return self._retriever.retrieve(query)

    def _build_context(self, results: list[dict[str, Any]]) -> str:
        """Build context string with [Page N] prefixes, within char limit."""
        parts: list[str] = []
        total = 0
        for r in results:
            meta = r.get("metadata", {}) or {}
            text = meta.get("text", "")
            if not text:
                continue
            page = meta.get("page_number")
            prefix = f"[Page {page}] " if page is not None else ""
            segment = prefix + text
            if total + len(segment) > self._context_max_chars:
                if not parts:
                    parts.append(segment[: self._context_max_chars])
                break
            parts.append(segment)
            total += len(segment)
        return "\n\n---\n\n".join(parts)

    def run(
        self,
        query: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline: retrieve → build prompt → call Ollama.

        Returns:
            {
                "answer": str,
                "sources": list[dict],  # per-chunk metadata (text, page_number, chunk_id, source_path)
                "similarity_scores": list[float],
            }
            If no relevant chunks are found, answer is NO_CONTEXT_MESSAGE and sources/similarity_scores are [].
        """
        query = (query or "").strip()
        if not query:
            return {
                "answer": "Please provide a non-empty query.",
                "sources": [],
                "similarity_scores": [],
            }

        results = self.retrieve(query)
        if not results:
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "sources": [],
                "similarity_scores": [],
            }

        formatted = _format_metric_dual_response(results)
        if formatted is not None:
            return {
                "answer": formatted,
                "sources": _format_sources(results),
                "similarity_scores": _format_similarity_scores(results),
            }

        context = self._build_context(results)
        if not context.strip():
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "sources": [],
                "similarity_scores": [],
            }

        user_message = build_rag_user_message(context, query)
        answer = self._llm.generate(
            user_message,
            system=self._system_prompt,
            temperature=temperature if temperature is not None else 0.1,
        )

        return {
            "answer": answer,
            "sources": _format_sources(results),
            "similarity_scores": _format_similarity_scores(results),
        }

    def query(self, question: str, *, temperature: float | None = None) -> str:
        """Run pipeline and return only the answer string (backward compatible)."""
        return self.run(question, temperature=temperature)["answer"]

    def query_with_sources(self, question: str, *, temperature: float | None = None) -> tuple[str, list[dict]]:
        """Run pipeline and return (answer, list of raw result dicts). Backward compatible."""
        out = self.run(question, temperature=temperature)
        return out["answer"], out["sources"]
