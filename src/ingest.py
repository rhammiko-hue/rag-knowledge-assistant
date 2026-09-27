"""Document loading and text chunking with overlap.

The corpus lives in data/ as one markdown file per document. Each document
is split into overlapping word-window chunks so that a fact spanning a
chunk boundary still appears whole in at least one chunk.
"""
import re
from pathlib import Path


def load_documents(data_dir):
    """Return [{'id', 'title', 'text'}] for every .md file in data_dir."""
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        # The first markdown heading is the document title.
        m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        title = m.group(1).strip() if m else path.stem
        # Strip markdown heading markers, ending each heading with a period so
        # sentence splitting sees clean prose ("## Foo Bar." not "## Foo Bar Baz.").
        text = re.sub(r"^#{1,6}\s+(.+)$", r"\1.", raw, flags=re.MULTILINE)
        docs.append({"id": path.stem, "title": title, "text": text})
    return docs


def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into word windows of chunk_size with `overlap` words shared
    between consecutive chunks. Overlap preserves context at boundaries."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def ingest(data_dir, chunk_size=200, overlap=50):
    """Load every document and return chunk records with metadata.

    Each record: {'doc_id', 'title', 'chunk_id', 'text'}.
    """
    records = []
    for doc in load_documents(data_dir):
        for i, ch in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            records.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "chunk_id": i,
                "text": ch,
            })
    return records
