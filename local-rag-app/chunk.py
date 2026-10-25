from ingest import Document
from models import Chunk


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Split text into overlapping, word-aligned character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    text = text.strip()
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk, source, start, end))

        if end == len(text):
            break
        start = end - overlap

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Convert an ingested document into chunks while preserving its source."""
    return chunk_text(
        document.text,
        source=document.source,
        chunk_size=chunk_size,
        overlap=overlap,
    )