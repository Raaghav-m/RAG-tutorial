from pathlib import Path

import pytest

from ingest import Document
from validate import validate_document


def test_validate_document_accepts_non_empty_document(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Useful text", encoding="utf-8")
    document = Document("Useful text", "notes.txt", str(path))

    validate_document(document)


def test_validate_document_rejects_empty_text(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    document = Document("   ", "empty.txt", str(path))

    with pytest.raises(ValueError, match="no extracted text"):
        validate_document(document)


def test_validate_document_rejects_large_text(tmp_path: Path):
    path = tmp_path / "large.txt"
    path.write_text("text", encoding="utf-8")
    document = Document("12345", "large.txt", str(path))

    with pytest.raises(ValueError, match="character limit"):
        validate_document(document, max_characters=4)