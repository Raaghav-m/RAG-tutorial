from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from models import Chunk


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


def load_embeddings(
    data_dir: Path,
) -> tuple[np.ndarray, list[Chunk], str]:
    embeddings = np.load(data_dir / "embeddings.npy")
    metadata = json.loads(
        (data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    chunks = [Chunk(**chunk) for chunk in metadata["chunks"]]
    return embeddings, chunks, metadata["model"]