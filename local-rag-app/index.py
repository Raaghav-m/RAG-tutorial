from dataclasses import asdict
import json
from pathlib import Path

import faiss
import numpy as np

from models import Chunk


def build_index(
    output_dir: Path,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """Build and persist a cosine-similarity FAISS index."""
    if len(embeddings) != len(chunks):
        raise ValueError("embeddings and chunks must have the same length")
    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("embeddings must be a non-empty two-dimensional array")

    vectors = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    np.save(output_dir / "embeddings.npy", vectors)

    metadata = {
        "model": model_name,
        "dimension": vectors.shape[1],
        "metric": "inner_product_on_normalized_embeddings",
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_index(data_dir: Path) -> tuple[faiss.Index, list[Chunk], str]:
    """Load a persisted FAISS index and its matching chunk metadata."""
    index = faiss.read_index(str(data_dir / "index.faiss"))
    metadata = json.loads(
        (data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    chunks = [Chunk(**chunk) for chunk in metadata["chunks"]]

    if index.ntotal != len(chunks):
        raise ValueError("index and metadata contain different numbers of chunks")
    if index.d != metadata["dimension"]:
        raise ValueError("index dimension does not match metadata")

    return index, chunks, metadata["model"]