#!/usr/bin/env python3
"""Interactive CLI for the RAG knowledge assistant.

Usage:
    python chat.py

On first run it ingests data/, chunks the documents, embeds them with
all-MiniLM-L6-v2, and saves a FAISS index under store/. Later runs reuse
the saved index. Type 'quit' to exit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingest import ingest
from src.embed import embed_texts
from src.store import build_index, save_store, load_store
from src.retrieve import retrieve
from src.generate import get_generator

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
STORE_DIR = BASE / "store"
STORE_PREFIX = str(STORE_DIR / "index")
TOP_K = 4


def ensure_index():
    """Load the saved index, or build it from data/ on first run."""
    faiss_path = STORE_DIR / "index.faiss"
    if faiss_path.exists():
        print("Loading saved index...")
        return load_store(STORE_PREFIX)
    print("First run: ingesting documents, embedding, building index (one-time)...")
    records = ingest(str(DATA_DIR))
    print(f"  {len(records)} chunks from documents in data/")
    embeddings = embed_texts([r["text"] for r in records])
    index = build_index(embeddings)
    STORE_DIR.mkdir(exist_ok=True)
    save_store(index, records, STORE_PREFIX)
    print("  index saved to store/")
    return index, records


def answer_question(question, index, records, generator, k=TOP_K, show_sources=True):
    retrieved = retrieve(question, index, records, k=k)
    result = generator.generate(question, retrieved)
    return result, retrieved


def main():
    index, records = ensure_index()
    generator = get_generator()
    print(f"\nRAG knowledge assistant ready ({type(generator).__name__}).")
    print("Ask about warehouse, inventory, freight, suppliers, TMS, cold chain,")
    print("returns, cycle counting, last-mile delivery, or customs. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        result, retrieved = answer_question(question, index, records, generator)
        print(f"\nAssistant:\n{result['answer']}")
        if retrieved:
            scores = ", ".join(f"{r['record']['doc_id']}:{r['score']:.2f}" for r in retrieved)
            print(f"\n(retrieved chunks: {scores})")
        print()


if __name__ == "__main__":
    main()
