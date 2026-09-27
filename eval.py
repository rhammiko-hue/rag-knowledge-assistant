#!/usr/bin/env python3
"""Evaluation for the RAG knowledge assistant.

Runs 10 hand-written question/answer pairs and reports:
  - hit@1 / hit@3 : fraction of questions whose gold document appears in
    the top-1 / top-3 retrieved chunks
  - key-phrase rate: fraction of generated answers containing the expected
    key phrase (case-insensitive)

Usage:
    python eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat import ensure_index, answer_question
from src.generate import ExtractiveGenerator

# (question, gold doc_id, key phrase expected in the answer)
QA_PAIRS = [
    ("What are the receiving dock's operating hours?",
     "doc_01_warehouse_receiving", "6:00 AM to 6:00 PM"),
    ("How is safety stock calculated?",
     "doc_02_inventory_policy", "average daily demand"),
    ("What invoice tolerance is allowed before manual review?",
     "doc_03_freight_audit", "2%"),
    ("What tax form must new suppliers submit?",
     "doc_04_supplier_onboarding", "W-9"),
    ("What is the target on-time pickup rate for carriers?",
     "doc_05_tms_optimization", "98%"),
    ("What temperature range is required for refrigerated goods?",
     "doc_06_cold_chain", "8 degrees Celsius"),
    ("How long is the customer return window?",
     "doc_07_returns_logistics", "30 days"),
    ("How often are A-class SKUs cycle counted?",
     "doc_08_cycle_counting", "weekly"),
    ("When must metro orders be placed for same-day delivery?",
     "doc_09_last_mile", "2:00 PM"),
    ("Which CBP form is required for US imports?",
     "doc_10_customs_compliance", "7501"),
]


def main():
    index, records = ensure_index()
    generator = ExtractiveGenerator()  # eval the offline default path

    hits_at_1 = hits_at_3 = key_phrase_ok = 0
    print(f"{'#':<3} {'question':<58} {'hit@1':<6} {'hit@3':<6} {'key-phrase'}")
    print("-" * 95)
    for i, (q, gold, phrase) in enumerate(QA_PAIRS, 1):
        result, retrieved = answer_question(q, index, records, generator, k=3)
        ranked_docs = [r["record"]["doc_id"] for r in retrieved]
        h1 = gold == ranked_docs[0] if ranked_docs else False
        h3 = gold in ranked_docs
        kp = phrase.lower() in result["answer"].lower()
        hits_at_1 += h1
        hits_at_3 += h3
        key_phrase_ok += kp
        short_q = (q[:55] + "...") if len(q) > 55 else q
        print(f"{i:<3} {short_q:<58} {str(h1):<6} {str(h3):<6} {kp}")
        if not (h3 and kp):
            print(f"      -> retrieved: {ranked_docs}, answer had phrase '{phrase}': {kp}")

    n = len(QA_PAIRS)
    print("-" * 95)
    print(f"hit@1:          {hits_at_1}/{n} = {hits_at_1 / n:.2f}")
    print(f"hit@3:          {hits_at_3}/{n} = {hits_at_3 / n:.2f}")
    print(f"key-phrase hit: {key_phrase_ok}/{n} = {key_phrase_ok / n:.2f}")


if __name__ == "__main__":
    main()
