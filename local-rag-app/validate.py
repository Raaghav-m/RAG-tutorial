from pathlib import Path

from ingest import Document, SUPPORTED_EXTENSIONS


def validate_document(
    document: Document,
    max_characters: int = 5_000_000,
) -> None:
    """Raise ValueError when an ingested document is not usable."""
    path = Path(document.path)
    if not path.is_file():
        raise ValueError(f"document file does not exist: {document.path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported document type: {path.suffix}")

    if not document.text.strip():
        raise ValueError("document contains no extracted text")

    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    if len(document.text) > max_characters:
        raise ValueError(
            f"document exceeds the {max_characters:,}-character limit"
        )