from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class Query:
    original: str
    rewritten: str


@dataclass
class GenerationResult:
    answer: str
    context: list[SearchResult]