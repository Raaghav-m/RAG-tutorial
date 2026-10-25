import numpy as np
import pytest

from rag_pipeline import Chunk, search_embeddings


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