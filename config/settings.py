"""Application settings and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

for d in (DATA_DIR, RAW_DOCS_DIR, PROCESSED_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SIZE_WORDS = 650
CHUNK_OVERLAP_WORDS = 80
SUPPORTED_EXTENSIONS = {".pdf"}

EMBEDDING_MODEL = "BAAI/bge-small-en"
EMBEDDING_DIM = 384

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 120.0
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 2048


def get_settings() -> dict[str, object]:
    """Return current settings as a dict (for dependency injection / overrides)."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_timeout": OLLAMA_TIMEOUT,
        "faiss_index_path": FAISS_INDEX_PATH,
        "chunks_metadata_path": CHUNKS_METADATA_PATH,
    }
