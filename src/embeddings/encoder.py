"""SentenceTransformers-based embedding encoder."""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "BAAI/bge-small-en"

_instance: "EmbeddingEncoder | None" = None


def get_encoder(model_name: str = DEFAULT_MODEL) -> "EmbeddingEncoder":
    """Return the shared encoder instance; loads the model once (singleton)."""
    global _instance
    if _instance is None:
        _instance = EmbeddingEncoder(model_name=model_name)
    return _instance


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors in place; supports 1D and 2D."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class EmbeddingEncoder:
    """Encode text to vectors using SentenceTransformers; use get_encoder() for singleton."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Loaded encoder %s (dim=%d)", model_name, self.dimension)

    @property
    def dimension(self) -> int:
        """Embedding dimension of the model."""
        return self._model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string. Returns L2-normalized 1D vector.

        Args:
            text: Query or sentence to embed.

        Returns:
            numpy array of shape (dimension,) and dtype float32.
        """
        vectors = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        out = vectors[0].astype(np.float32)
        if not (np.linalg.norm(out) > 0):
            return out
        return _normalize_l2(out.reshape(1, -1))[0]

    def embed_chunks(self, texts: List[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of text chunks. Returns L2-normalized 2D array.

        Args:
            texts: List of chunk strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        out = vectors.astype(np.float32)
        return _normalize_l2(out)

    def encode(self, texts: List[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts; returns normalized (n, dim) array."""
        return self.embed_chunks(texts, batch_size=batch_size, show_progress=show_progress)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string. Returns normalized 1D array."""
        return self.embed_query(text)
