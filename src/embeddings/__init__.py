"""Embeddings via SentenceTransformers."""

from src.embeddings.encoder import (
    DEFAULT_MODEL,
    EmbeddingEncoder,
    get_encoder,
)


def embed_query(text: str, *, model_name: str = DEFAULT_MODEL):
    """Embed a single query using the shared model. Returns normalized 1D numpy array."""
    return get_encoder(model_name).embed_query(text)


def embed_chunks(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
):
    """Embed a list of chunks using the shared model. Returns normalized (n, dim) numpy array."""
    return get_encoder(model_name).embed_chunks(
        texts, batch_size=batch_size, show_progress=show_progress
    )


__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingEncoder",
    "get_encoder",
    "embed_query",
    "embed_chunks",
]
