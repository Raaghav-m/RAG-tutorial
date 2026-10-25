import numpy as np
import faiss

from embed import embed_chunks
from models import Chunk, SearchResult


def search_embeddings(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    """Return chunks with the highest cosine-similarity scores."""
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


def search_index(
    query: str,
    index: faiss.Index,
    chunks: list[Chunk],
    model_name: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed a query and search the persisted FAISS index."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if index.ntotal == 0:
        return []
    if index.ntotal != len(chunks):
        raise ValueError("index and chunks must have the same length")

    query_embeddings = embed_chunks(
        [Chunk(query, source="query", start=0, end=len(query))],
        model_name=model_name,
    )
    if query_embeddings.shape != (1, index.d):
        raise ValueError(
            f"query embedding dimension {query_embeddings.shape} does not "
            f"match index dimension {index.d}"
        )

    query_embedding = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, indices = index.search(query_embedding, min(top_k, index.ntotal))
    return [
        SearchResult(chunks[index], float(score))
        for score, index in zip(scores[0], indices[0])
        if index >= 0
    ]