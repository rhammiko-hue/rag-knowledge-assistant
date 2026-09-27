"""Answer generation.

`Generator` is the single interface every answer strategy implements:
    generate(query, retrieved) -> {"answer": str, "citations": [str]}

The default is `ExtractiveGenerator`, which works fully offline: it scores
sentences from the retrieved chunks by keyword overlap with the question
and composes the answer from the best ones, citing the source documents.

`OpenAIGenerator` shows how an LLM plugs into the same interface later:
set OPENAI_API_KEY and it synthesizes the answer from the retrieved
context instead. Nothing else in the pipeline changes.
"""
import os
import re
from abc import ABC, abstractmethod

STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been by as at "
    "from that this these those it its into what when where which who whom "
    "how why does do did can could should would will shall may might must "
    "have has had having there their them they he she we you your our ours "
    "if then than so such no not but about over under again once here".split()
)


class Generator(ABC):
    @abstractmethod
    def generate(self, query, retrieved):
        """Build an answer from retrieved chunks.

        Args:
            query: the user's question (str).
            retrieved: list of {"record", "score"} from retrieve().
        Returns:
            {"answer": str, "citations": [document titles used]}.
        """


def _sentences(text):
    # Skip short fragments (e.g. bare section headings) — they carry no
    # standalone meaning as answer sentences.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]


def _keywords(query):
    words = re.findall(r"[A-Za-z0-9%]+", query.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


class ExtractiveGenerator(Generator):
    """Offline answer composer: picks the highest-overlap sentences from the
    retrieved chunks and presents them with source citations."""

    def __init__(self, max_sentences=3):
        self.max_sentences = max_sentences

    def generate(self, query, retrieved):
        keywords = _keywords(query)
        candidates = []  # (hits, score, order, sentence, title)
        order = 0
        for r in retrieved:
            rec = r["record"]
            for sent in _sentences(rec["text"]):
                sent_words = set(re.findall(r"[A-Za-z0-9%]+", sent.lower()))
                hits = len(keywords & sent_words)
                if hits > 0:
                    candidates.append((hits, r["score"], order, sent, rec["title"]))
                    order += 1
        # Best keyword overlap first, chunk relevance as tiebreak.
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))

        picked, seen_titles = [], []
        for hits, _, _, sent, title in candidates:
            if sent not in {p[0] for p in picked}:
                picked.append((sent, title))
                if title not in seen_titles:
                    seen_titles.append(title)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            return {
                "answer": "I couldn't find relevant information in the knowledge base for that question.",
                "citations": [],
            }
        lines = ["Based on the knowledge base:\n"]
        for sent, title in picked:
            lines.append(f"- {sent} [Source: {title}]")
        return {"answer": "\n".join(lines), "citations": seen_titles}


class OpenAIGenerator(Generator):
    """LLM-backed generator behind the same interface.

    Requires the OPENAI_API_KEY environment variable. Raises a clear error
    otherwise, so the offline path stays the default.
    """

    def __init__(self, model="gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAIGenerator.")
        from openai import OpenAI  # imported lazily: openai is an optional dep

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, query, retrieved):
        context = "\n\n".join(
            f"[{r['record']['title']}]\n{r['record']['text']}" for r in retrieved
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using ONLY the provided context. "
                    "Cite the document title for every fact you state.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        citations = [r["record"]["title"] for r in retrieved]
        return {"answer": answer, "citations": citations}


def get_generator():
    """Factory: OpenAI-backed if an API key is configured, otherwise the
    offline extractive generator."""
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerator()
    return ExtractiveGenerator()
