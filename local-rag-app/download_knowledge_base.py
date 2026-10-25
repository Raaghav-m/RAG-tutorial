import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "framolfese/2WikiMultihopQA"
SPLIT = "train"
ROWS = 100
OUTPUT_PATH = Path("documents/2wikimultihop_contexts.txt")


def download_rows(offset: int = 0, length: int = ROWS) -> list[dict]:
    params = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urlopen(url) as response:
        payload = json.load(response)
    return [row["row"] for row in payload["rows"]]


def build_context_corpus(rows: list[dict]) -> str:
    sections: list[str] = []
    seen_titles: set[str] = set()

    for row in rows:
        context = row.get("context", {})
        for title, sentences in zip(
            context.get("title", []), context.get("sentences", [])
        ):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sections.append(f"TITLE: {title}\n{''.join(sentences)}")

    return "\n\n".join(sections) + "\n"


def main() -> None:
    rows = download_rows()
    corpus = build_context_corpus(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(corpus, encoding="utf-8")
    print(f"Downloaded {len(rows)} dataset examples")
    print(f"Created {len(corpus):,} characters of context")
    print(f"Saved knowledge base to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()