import numpy as np
from sentence_transformers import SentenceTransformer

from models import Chunk


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load the local embedding model, downloading it only on first use."""
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Create normalized local embeddings for chunks."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)