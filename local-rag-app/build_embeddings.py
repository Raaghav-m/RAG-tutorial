from pathlib import Path
import sys

from chunk import chunk_document
from embed import embed_chunks
from ingest import ingest_file
from index import build_index
from validate import validate_document


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../sample.txt")
    output_dir = Path("data")
    document = ingest_file(source_path)
    validate_document(document)
    chunks = chunk_document(document)
    embeddings = embed_chunks(chunks, model_name=MODEL_NAME)
    build_index(output_dir, chunks, embeddings, MODEL_NAME)
    print(f"Created {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved FAISS index to {output_dir / 'index.faiss'}")


if __name__ == "__main__":
    main()