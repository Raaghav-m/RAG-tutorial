import pytest

from cite import cite_results, format_citation
from models import Chunk, SearchResult


def test_cite_results_preserves_source_offsets_and_score():
    results = [
        SearchResult(Chunk("text", "sample.txt", 10, 14), 0.8765),
    ]

    citations = cite_results(results)

    assert citations[0].source == "sample.txt"
    assert citations[0].start == 10
    assert citations[0].end == 14
    assert citations[0].score == pytest.approx(0.8765)


def test_format_citation_is_readable():
    citation = cite_results(
        [SearchResult(Chunk("text", "sample.txt", 10, 14), 0.8765)]
    )[0]

    assert format_citation(1, citation) == (
        "[1] sample.txt (characters 10-14, score 0.8765)"
    )