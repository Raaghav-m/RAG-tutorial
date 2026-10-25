import json

import faiss
import numpy as np
import pytest

from index import build_index, load_index
from models import Chunk


def test_build_index_persists_vectors_and_metadata(tmp_path):
    chunks = [
        Chunk("first", "sample.txt", 0, 5),
        Chunk("second", "sample.txt", 6, 12),
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    build_index(tmp_path, chunks, embeddings, "test-model")
    index, loaded_chunks, model_name = load_index(tmp_path)

    assert isinstance(index, faiss.Index)
    assert index.ntotal == 2
    assert index.d == 2
    assert loaded_chunks == chunks
    assert model_name == "test-model"
    assert np.load(tmp_path / "embeddings.npy").shape == (2, 2)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["metric"] == "inner_product_on_normalized_embeddings"


def test_build_index_rejects_mismatched_vectors(tmp_path):
    chunks = [Chunk("only", "sample.txt", 0, 4)]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="same length"):
        build_index(tmp_path, chunks, embeddings, "test-model")