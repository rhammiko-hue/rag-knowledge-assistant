"""Embeddings via sentence-transformers (all-MiniLM-L6-v2).

Embeddings are L2-normalized, so cosine similarity == inner product.
That lets us use a plain FAISS IndexFlatIP for exact nearest-neighbor
search. The model is loaded lazily and cached: loading it twice would
waste ~90 MB of RAM.
"""
import numpy as np


def _sanitize_no_proxy():
    """Drop bracketed IPv6 entries (e.g. "[::1]") from no_proxy/NO_PROXY.

    Why: huggingface_hub builds its HTTP client with httpx, and httpx
    raises `InvalidURL: Invalid port` when parsing a bracketed IPv6
    literal in no_proxy. The unbracketed equivalents (e.g. "::1") are left
    untouched, so proxy-bypass behavior is preserved. This only affects
    the current process's environment.
    """
    import os

    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var)
        if val and "[" in val:
            kept = [p for p in val.split(",")
                    if not (p.strip().startswith("[") and p.strip().endswith("]"))]
            os.environ[var] = ",".join(kept)


_sanitize_no_proxy()

_model = None


def get_model(name="all-MiniLM-L6-v2"):
    """Load (once) and return the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts, model=None):
    """Embed a list of strings -> float32 numpy array, rows L2-normalized."""
    model = model or get_model()
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)
