from dataclasses import dataclass

from models import SearchResult


@dataclass
class Citation:
    source: str
    start: int
    end: int
    score: float


def cite_results(results: list[SearchResult]) -> list[Citation]:
    """Turn retrieved chunks into source citations."""
    return [
        Citation(
            source=result.chunk.source,
            start=result.chunk.start,
            end=result.chunk.end,
            score=result.score,
        )
        for result in results
    ]


def format_citation(rank: int, citation: Citation) -> str:
    """Format one citation for terminal output."""
    return (
        f"[{rank}] {citation.source} "
        f"(characters {citation.start}-{citation.end}, "
        f"score {citation.score:.4f})"
    )