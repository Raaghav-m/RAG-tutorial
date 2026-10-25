import pytest

from rewrite import rewrite_query


def test_rewrite_query_normalizes_whitespace_and_preserves_original():
    class OfflineClient:
        def chat(self, **_kwargs):
            raise OSError("Ollama is not running")

    query = rewrite_query(
        "  What   is\ncosine similarity?  ",
        client=OfflineClient(),
    )

    assert query.original == "  What   is\ncosine similarity?  "
    assert query.rewritten == "What is cosine similarity?"


def test_rewrite_query_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        rewrite_query(" \n\t ")


def test_rewrite_query_uses_ollama_json_response():
    class FakeClient:
        def chat(self, **_kwargs):
            return {"message": {"content": '{"rewritten":"Who directed Move?"}'}}

    query = rewrite_query("What about the director of Move?", client=FakeClient())

    assert query.original == "What about the director of Move?"
    assert query.rewritten == "Who directed Move?"


def test_rewrite_query_falls_back_when_ollama_is_unavailable():
    class OfflineClient:
        def chat(self, **_kwargs):
            raise OSError("Ollama is not running")

    query = rewrite_query("  What   is RAG?  ", client=OfflineClient())

    assert query.rewritten == "What is RAG?"