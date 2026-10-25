from pathlib import Path
import sys

from rag_pipeline import chunk_text, embed_chunks, save_embeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../sample.txt")
    output_dir = Path("data")
    text = source_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, source=source_path.name)
    embeddings = embed_chunks(chunks, model_name=MODEL_NAME)
    save_embeddings(output_dir, chunks, embeddings, MODEL_NAME)
    print(f"Created {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved artifacts to {output_dir}/")


if __name__ == "__main__":
    main()