from pathlib import Path
import sys

from cite import cite_results, format_citation
from generate import generate_answer
from index import load_index
from retrieve import search_index
from rewrite import rewrite_query


def main() -> None:
    raw_query = " ".join(sys.argv[1:])
    if not raw_query.strip():
        raise SystemExit("Usage: python search.py <question>")

    query = rewrite_query(raw_query)
    index, chunks, model_name = load_index(Path("data"))
    results = search_index(query.rewritten, index, chunks, model_name)
    generated = generate_answer(query.original, results)
    citations = cite_results(generated.context)

    print(generated.answer)
    print("\nCitations:")
    for rank, citation in enumerate(citations, start=1):
        print(format_citation(rank, citation))

    print("\nRetrieved context:")
    for result in generated.context:
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()