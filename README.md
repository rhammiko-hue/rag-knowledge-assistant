# RAG Knowledge Assistant (Supply Chain)

A retrieval-augmented Q&A chatbot over a small corpus of supply-chain /
logistics documents. Ask it about warehouse receiving, inventory policy,
freight audit, supplier onboarding, TMS optimization, cold chain, returns,
cycle counting, last-mile delivery, or customs compliance — it retrieves
the relevant passages and composes a cited answer. **Fully offline** once
the embedding model is downloaded (no API key needed).

## Architecture

```
                    ┌─────────────┐
                    │  data/*.md  │  10 hand-written supply-chain docs
                    └──────┬──────┘
                           │ ingest.py — load + clean + chunk
                           │ (200-word windows, 50-word overlap)
                           ▼
                    ┌─────────────┐
                    │  chunks     │  {doc_id, title, chunk_id, text}
                    └──────┬──────┘
                           │ embed.py — all-MiniLM-L6-v2
                           │ (L2-normalized, 384-dim)
                           ▼
                    ┌─────────────┐
                    │ FAISS index │  IndexFlatIP (exact cosine search)
                    │  store/     │  saved to disk after first build
                    └──────┬──────┘
                           │
User question ──▶ retrieve.py (embed query, top-k=4 search)
                           │
                           ▼
                  ┌────────────────┐
                  │   Generator    │  single interface:
                  │  ┌──────────┐  │  generate(query, chunks)
                  │  │Extractive│◀─┼─ default, offline: picks best
                  │  │(default) │  │  sentences by keyword overlap,
                  │  └──────────┘  │  cites source documents
                  │  ┌──────────┐  │
                  │  │ OpenAI   │◀─┼─ used automatically if
                  │  │(optional)│  │  OPENAI_API_KEY is set
                  │  └──────────┘  │
                  └────────────────┘
                           │
                           ▼
                 Cited answer + retrieval scores
```

## Install & run

```bash
pip install -r requirements.txt   # faiss-cpu, sentence-transformers, numpy
python chat.py                    # interactive CLI (builds index on first run)
python eval.py                    # runs the 10-question eval set
```

First run downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face and
builds `store/`; later runs reuse it. To use the LLM path instead of the
extractive fallback: `export OPENAI_API_KEY=...` and run `chat.py` —
`get_generator()` swaps in `OpenAIGenerator` behind the same interface.

## Example session (real run)

```
You: What documents does a new supplier need to submit?

Assistant:
Based on the knowledge base:

- New suppliers must submit a completed W-9 (or W-8BEN for foreign entities),
  a certificate of insurance meeting our minimum coverage limits, banking
  details for ACH payments, and a signed supplier code of conduct.
  [Source: Supplier Onboarding]
- Once approved, the supplier record is created in the ERP system with payment
  terms, lead times, and approved ship-from locations. [Source: Supplier Onboarding]
- New suppliers are reviewed after the first 90 days. [Source: Supplier Onboarding]

(retrieved chunks: doc_04_supplier_onboarding:0.61, doc_10_customs_compliance:0.47,
 doc_03_freight_audit:0.25, doc_01_warehouse_receiving:0.24)

You: What is the return window for customers?

Assistant:
Based on the knowledge base:

- Customers have 30 days from the delivery date to return unused items in their
  original packaging. [Source: Returns and Reverse Logistics]
- Refunds are issued to the original payment method within 5 business days of the
  return being graded as resellable. [Source: Returns and Reverse Logistics]
```

## Evaluation (measured, not estimated)

`python eval.py` runs 10 hand-written Q/A pairs (one per document) and
reports, measured on 2026-09-27:

| metric | score |
|---|---|
| hit@1 (gold doc is top retrieved chunk) | **9/10 = 0.90** |
| hit@3 (gold doc in top 3) | **10/10 = 1.00** |
| key-phrase hit (expected fact appears in generated answer) | **10/10 = 1.00** |

The one hit@1 miss: *"What is the target on-time pickup rate for carriers?"*
retrieved the freight-audit doc's "on-time pickup" mention first; the correct
TMS doc (98%) ranked in the top 3 and the answer still contained the key
phrase. Note the eval set is small and hand-written against this corpus, so
treat it as a regression smoke test, not a benchmark.

## Limitations (interview-honest)

- **Extractive, not generative, by default.** Without an LLM API key the
  "answer" is composed of the highest-overlapping sentences from retrieved
  chunks — fluent and cited, but it cannot paraphrase or reason across
  documents. Set `OPENAI_API_KEY` for abstractive answers via the same
  `Generator` interface.
- **Tiny corpus, exact search.** 10 documents / 10 chunks with
  `IndexFlatIP` (exact search). Production scale would need an approximate
  index (IVF/PQ) and re-ranking.
- **No conversation memory.** Each question is answered independently; there
  is no multi-turn context or follow-up handling.
- **Keyword-based sentence scoring** can surface a partially relevant
  sentence alongside the right one (see the hit@1 miss above).
- Chunking uses fixed word windows (200 words, 50 overlap) rather than
  semantic/structural splitting.

## Files

```
rag-knowledge-assistant/
├── data/               # 10 hand-written supply-chain docs (.md)
├── src/
│   ├── ingest.py       # load, clean markdown, chunk with overlap
│   ├── embed.py        # sentence-transformers embeddings (cached model)
│   ├── store.py        # FAISS build/save/load
│   ├── retrieve.py     # top-k retrieval
│   └── generate.py     # Generator interface + Extractive/OpenAI impls
├── chat.py             # interactive CLI
├── eval.py             # 10-QA eval: hit@1/hit@3/key-phrase
├── requirements.txt
└── README.md
```
