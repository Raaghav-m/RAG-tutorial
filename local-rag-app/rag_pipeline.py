from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Split text into overlapping, word-aligned character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    text = text.strip()
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk, source, start, end))

        if end == len(text):
            break
        start = end - overlap

    return chunks


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Create normalized local embeddings for chunks."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def search_embeddings(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    """Return the chunks with the highest cosine-similarity scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(embeddings) != len(chunks):
        raise ValueError("embeddings and chunks must have the same length")
    if not len(chunks):
        return []

    scores = embeddings @ query_embedding
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(chunks[index], float(scores[index]))
        for index in best_indices
    ]


def search_chunks(
    query: str,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed a query locally and return its most similar chunks."""
    query_embedding = embed_chunks(
        [Chunk(query, source="query", start=0, end=len(query))],
        model_name=model_name,
    )[0]
    return search_embeddings(query_embedding, embeddings, chunks, top_k=top_k)


def save_embeddings(
    output_dir: Path,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    metadata = {
        "model": model_name,
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )