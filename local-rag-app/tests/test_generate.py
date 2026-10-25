import pytest

from generate import build_prompt, generate_answer
from models import Chunk, SearchResult


def test_build_prompt_contains_question_and_sources():
    results = [
        SearchResult(Chunk("Cosine text", "sample.txt", 0, 11), 0.9),
    ]

    prompt = build_prompt("What is cosine similarity?", results)

    assert "What is cosine similarity?" in prompt
    assert "SOURCE: sample.txt" in prompt
    assert "Cosine text" in prompt
    assert "only the provided context" in prompt


def test_generate_answer_parses_structured_ollama_response():
    class FakeClient:
        def chat(self, **kwargs):
            assert kwargs["options"]["temperature"] == 0
            return {"message": {"content": '{"answer":"It measures vector angle."}'}}

    result = generate_answer(
        "What is cosine similarity?",
        [SearchResult(Chunk("Cosine text", "sample.txt", 0, 11), 0.9)],
        client=FakeClient(),
    )

    assert result.answer == "It measures vector angle."
    assert len(result.context) == 1


def test_generate_answer_handles_no_context():
    result = generate_answer("Unknown?", [], client=object())

    assert "do not know" in result.answer
    assert result.context == []


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_answer(" ", [])