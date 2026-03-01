<<<<<<< HEAD
# Local RAG System

A modular, production-ready **Retrieval-Augmented Generation** pipeline that runs entirely on your machine. No LangChain.

**Stack:** PyMuPDF · SentenceTransformers · FAISS · Ollama · Streamlit

## Structure

```
├── config/           # Settings and paths
├── data/
│   ├── raw_documents/   # Put your PDFs here
│   ├── processed/
│   └── index/           # FAISS index + metadata
├── src/
│   ├── ingestion/    # PDF load + chunk (PyMuPDF)
│   ├── embeddings/   # SentenceTransformers
│   ├── vector_store/ # FAISS
│   ├── llm/          # Ollama client
│   ├── rag/          # Retrieve + generate pipeline
│   └── app/          # Streamlit UI
├── scripts/
│   ├── build_index.py   # Ingest PDFs → build FAISS index
│   └── run_app.py       # Start Streamlit
├── tests/
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

2. **Install and run Ollama** (for the LLM)

   - Download from [ollama.ai](https://ollama.ai) and run:
     ```bash
     ollama pull llama3.2
     ```
   - Adjust `config/settings.py` if you use another model or port.

3. **Add documents**

   - Place PDFs in `data/raw_documents/`.

4. **Build the index**

   ```bash
   python scripts/build_index.py
   ```

5. **Run the app**

   ```bash
   python scripts/run_app.py
   # or: streamlit run src/app/main.py
   ```

   Open the URL shown in the terminal (default `http://localhost:8501`).

## Configuration

Edit `config/settings.py`:

- **Chunking:** `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Embeddings:** `EMBEDDING_MODEL` (e.g. `all-MiniLM-L6-v2`)
- **Ollama:** `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- **Paths:** `RAW_DOCS_DIR`, `INDEX_DIR`, etc.

## Usage as a library

```python
from config.settings import FAISS_INDEX_PATH, CHUNKS_METADATA_PATH, EMBEDDING_MODEL, EMBEDDING_DIM
from src.embeddings import EmbeddingEncoder
from src.llm import OllamaClient
from src.rag import RAGPipeline
from src.vector_store import FAISSStore

encoder = EmbeddingEncoder(EMBEDDING_MODEL)
store = FAISSStore(dimension=EMBEDDING_DIM)
store.load(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
llm = OllamaClient()
rag = RAGPipeline(encoder=encoder, store=store, llm=llm, top_k=5)
answer = rag.query("Your question here")
```

## Scaling

- **More documents:** Add PDFs and re-run `build_index.py`. For very large corpora, consider FAISS IVF or HNSW (replace `IndexFlatL2` in `faiss_store.py`).
- **Other file types:** Extend `src/ingestion/loader.py` and add new loaders; keep using the same chunker and pipeline.
- **Other embedders/LLMs:** Swap `EmbeddingEncoder` and `OllamaClient` for your own implementations; the RAG pipeline stays the same.

## License

MIT
=======
# RAG-PROJECT
>>>>>>> 99436c9b89a56999b41be884682a27ffa7d1bdcb
