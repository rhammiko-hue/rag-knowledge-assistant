"""Top-k retrieval: embed the query, search the FAISS index."""
from src.embed import embed_texts


def retrieve(query, index, records, k=4):
    """Return the k most similar chunks as [{'record', 'score'}], best first.

    `score` is cosine similarity in [-1, 1]; higher is more relevant.
    """
    q = embed_texts([query])
    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:  # FAISS returns -1 when fewer than k vectors exist
            continue
        results.append({"record": records[int(idx)], "score": float(score)})
    return results
