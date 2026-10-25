import numpy as np
import pytest
import faiss

from index import build_index, load_index
from rag_pipeline import Chunk, search_embeddings
from retrieve import search_index


def test_search_embeddings_returns_highest_scores_first():
    chunks = [
        Chunk("low", "sample.txt", 0, 3),
        Chunk("high", "sample.txt", 4, 8),
        Chunk("middle", "sample.txt", 9, 15),
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )

    results = search_embeddings(np.array([0.0, 1.0]), embeddings, chunks, top_k=2)

    assert [result.chunk.text for result in results] == ["high", "middle"]
    assert results[0].score == pytest.approx(1.0)


def test_search_embeddings_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        search_embeddings(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]], dtype=np.float32),
            [],
        )


def test_search_index_returns_faiss_matches(tmp_path):
    chunks = [
        Chunk("low", "sample.txt", 0, 3),
        Chunk("high", "sample.txt", 4, 8),
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    build_index(tmp_path, chunks, embeddings, "test-model")
    index, loaded_chunks, model_name = load_index(tmp_path)

    class FakeModel:
        def encode(self, *_args, **_kwargs):
            return np.array([[0.0, 1.0]], dtype=np.float32)

    original = __import__("retrieve").embed_chunks
    __import__("retrieve").embed_chunks = lambda *_args, **_kwargs: FakeModel().encode()
    try:
        results = search_index("query", index, loaded_chunks, model_name, top_k=1)
    finally:
        __import__("retrieve").embed_chunks = original

    assert results[0].chunk.text == "high"
    assert results[0].score == pytest.approx(1.0)


def test_search_index_rejects_query_dimension_mismatch(tmp_path):
    chunks = [Chunk("text", "sample.txt", 0, 4)]
    build_index(
        tmp_path,
        chunks,
        np.array([[1.0, 0.0]], dtype=np.float32),
        "test-model",
    )
    index, loaded_chunks, model_name = load_index(tmp_path)

    retrieve_module = __import__("retrieve")
    original = retrieve_module.embed_chunks
    retrieve_module.embed_chunks = lambda *_args, **_kwargs: np.array(
        [[1.0, 0.0, 0.0]], dtype=np.float32
    )
    try:
        with pytest.raises(ValueError, match="does not match index dimension"):
            search_index("query", index, loaded_chunks, model_name)
    finally:
        retrieve_module.embed_chunks = original