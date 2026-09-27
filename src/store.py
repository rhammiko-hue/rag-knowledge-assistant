"""FAISS vector store: build, save, load.

We use IndexFlatIP (exact inner-product search). Because embeddings are
L2-normalized, inner product == cosine similarity, so higher score means
more semantically similar. Flat indexes are exact (no approximation) and
plenty fast for a small corpus; for millions of vectors you would switch
to an IVF/PQ index.
"""
import json
import faiss
import numpy as np


def build_index(embeddings):
    """Build an exact inner-product index from a (n, dim) float32 array."""
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_store(index, records, path_prefix):
    """Persist the FAISS index and the chunk metadata side by side."""
    faiss.write_index(index, path_prefix + ".faiss")
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def load_store(path_prefix):
    """Load a previously saved index + chunk metadata."""
    index = faiss.read_index(path_prefix + ".faiss")
    with open(path_prefix + ".json", encoding="utf-8") as f:
        records = json.load(f)
    return index, records
