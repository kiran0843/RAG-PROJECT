"""FAISS vector store: IndexFlatIP, save/load, top-k search."""

import json
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np


class FAISSStore:
    """Vector store using FAISS IndexFlatIP (cosine similarity on L2-normalized vectors)."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add embeddings and optional metadata to the index.

        Args:
            embeddings: Array of shape (n_vectors, dimension), dtype float32 preferred.
            metadata: Optional list of n_vectors dicts; if None, empty dicts are stored.
        """
        if embeddings.size == 0:
            return
        if embeddings.ndim == 1:
            embeddings = np.asarray(embeddings, dtype=np.float32).reshape(1, -1)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)
        n = len(embeddings)
        if metadata is not None and len(metadata) == n:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{}] * n)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search for the top-k nearest vectors by inner product (cosine similarity
        when vectors are L2-normalized).

        Args:
            query_embedding: Query vector, shape (dimension,) or (1, dimension).
            top_k: Number of neighbors to return (default 3).

        Returns:
            List of dicts, each with:
            - score: similarity score in (0, 1], higher = more similar.
            - distance: L2 distance (lower = more similar).
            - index: index of the vector in the store.
            - metadata: stored metadata for that vector.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        if k <= 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dimension:
            raise ValueError(f"Query dimension {q.shape[1]} != store dimension {self._dimension}")

        scores, indices = self._index.search(q, k)
        results: list[dict[str, Any]] = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx < 0:
                continue
            score = float(scores[0][i])
            results.append({
                "score": score,
                "distance": 1.0 - score,
                "index": idx,
                "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
            })
        return results

    def save(self, index_path: Path | str, metadata_path: Path | str | None = None) -> None:
        """
        Save the FAISS index and optionally metadata to disk.

        Args:
            index_path: Path for the .index file.
            metadata_path: Optional path for a JSON file with metadata list.
        """
        if self._index is None:
            raise RuntimeError("Cannot save: index is empty")
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        if metadata_path is not None and self._metadata:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    def load(self, index_path: Path | str, metadata_path: Path | str | None = None) -> None:
        """
        Load the FAISS index and optionally metadata from disk.

        Args:
            index_path: Path to the .index file.
            metadata_path: Optional path to the metadata JSON file.
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self._index = faiss.read_index(str(index_path))
        self._metadata = []
        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)

    def filter_by_metadata(
        self,
        predicate: Callable[[dict[str, Any]], bool],
    ) -> list[dict[str, Any]]:
        """
        Return chunks whose metadata matches the predicate. No vector search.
        Returns same format as search().
        """
        if not self._metadata:
            return []
        results: list[dict[str, Any]] = []
        for i, meta in enumerate(self._metadata):
            if predicate(meta):
                results.append({
                    "score": 1.0,
                    "distance": 0.0,
                    "index": i,
                    "metadata": meta,
                })
        return results

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index is not None else 0
