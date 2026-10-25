import json
from pathlib import Path
import sys

import numpy as np

from rag_pipeline import Chunk, search_chunks


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit("Usage: python search.py <question>")

    data_dir = Path("data")
    embeddings = np.load(data_dir / "embeddings.npy")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**chunk) for chunk in metadata["chunks"]]
    results = search_chunks(query, embeddings, chunks, metadata["model"])

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.4f} source={result.chunk.source}")
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()