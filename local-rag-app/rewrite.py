import json
import re
from typing import Any

import ollama

from models import Query


DEFAULT_REWRITE_MODEL = "llama3.1:8b"

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
}


def _fallback_query(text: str) -> Query:
    original = text
    rewritten = re.sub(r"\s+", " ", text).strip()
    if not rewritten:
        raise ValueError("query cannot be empty")
    return Query(original=original, rewritten=rewritten)


def rewrite_query(
    text: str,
    model_name: str = DEFAULT_REWRITE_MODEL,
    client: Any = ollama,
    use_fallback: bool = True,
) -> Query:
    """Rewrite a query with Ollama, falling back to normalization if needed."""
    fallback = _fallback_query(text)
    prompt = (
        "Rewrite the user's question into one precise, standalone search query. "
        "Preserve names, dates, and the original meaning. Do not answer the "
        "question. Return only JSON matching the requested schema.\n\n"
        f"User question: {fallback.rewritten}"
    )

    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format=REWRITE_SCHEMA,
            options={"temperature": 0},
        )
        content = response["message"]["content"]
        rewritten = json.loads(content)["rewritten"]
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if rewritten:
            return Query(original=text, rewritten=rewritten)
    except Exception:
        if not use_fallback:
            raise

    return fallback