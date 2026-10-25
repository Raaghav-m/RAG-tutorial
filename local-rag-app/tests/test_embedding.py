import numpy as np
import pytest

from embed import embed_chunks
from models import Chunk


def test_embed_chunks_returns_normalized_float32_vectors():
    chunks = [Chunk("A local retrieval system", "sample.txt", 0, 24)]

    embeddings = embed_chunks(chunks, batch_size=1)

    assert embeddings.shape == (1, 384)
    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0, abs=1e-5)


def test_embed_chunks_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        embed_chunks(
            [Chunk("text", "sample.txt", 0, 4)],
            batch_size=0,
        )