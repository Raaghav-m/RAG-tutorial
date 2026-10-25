import pytest

from chunk import chunk_document
from ingest import Document
from rag_pipeline import chunk_text


def test_chunking_keeps_source_and_covers_text():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, source="sample.txt", chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(chunk.source == "sample.txt" for chunk in chunks)
    assert chunks[0].text.startswith("one")
    assert chunks[-1].text.endswith("ten")


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("text", source="sample.txt", chunk_size=10, overlap=10)


def test_chunk_document_preserves_ingested_source():
    document = Document(
        text="one two three four five six",
        source="sample.txt",
        path="sample.txt",
    )

    chunks = chunk_document(document, chunk_size=20, overlap=5)

    assert chunks
    assert all(chunk.source == document.source for chunk in chunks)