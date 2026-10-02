"""Context assembly and labelled retrieval evaluation; no implicit model calls."""
from dataclasses import asdict, dataclass
from math import log2
from pathlib import Path
from .chunking import chunk_markdown
from .retrieval import Embedder, Hit, SearchIndex


def ingest(path: str | Path, max_chars: int = 1200, overlap: int = 120):
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*")
             if p.is_file() and not p.is_symlink() and p.suffix.lower() in {".md", ".txt"})
    chunks = []
    for file in files:
        source = file.name if root.is_file() else file.relative_to(root).as_posix()
        chunks.extend(chunk_markdown(file.read_text(encoding="utf-8"), max_chars, overlap, source))
    return chunks


@dataclass(frozen=True)
class Context:
    text: str
    citations: list[dict]

    def as_dict(self):
        return asdict(self)


def build_context(hits: list[Hit], max_chars: int = 6000) -> Context:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    parts, citations, seen = [], [], set()
    used = 0
    for hit in hits:
        c = hit.chunk
        if c.id in seen:
            continue
        seen.add(c.id)
        label = len(citations) + 1
        header = f"[{label}] {c.source}" + (f" | {c.heading}" if c.heading else "") + "\n"
        available = max_chars - used - len(header) - (2 if parts else 0)
        if available < 1:
            continue
        body = c.text[:available]
        part = header + body
        used += len(part) + (2 if parts else 0)
        parts.append(part)
        citations.append({"label": label, "chunk_id": c.id, "source": c.source,
                          "start": c.start, "end": c.start + len(body),
                          "truncated": len(body) < len(c.text)})
    return Context("\n\n".join(parts), citations)


def evaluate(index: SearchIndex, cases: list[dict], k: int = 5,
             embedder: Embedder | None = None) -> dict:
    """Macro-average source-level Recall@k, MRR@k and binary nDCG@k.

    Multiple chunks from one source count once; these are retrieval metrics,
    not claims about generated answer correctness.
    """
    if not cases or k < 1:
        raise ValueError("provide nonempty evaluation cases and k > 0")
    rows = []
    for case in cases:
        relevant = set(case["relevant_sources"])
        if not relevant:
            raise ValueError("each case needs at least one relevant source")
        hits = index.search(case["query"], k, embedder=embedder)
        ranked = list(dict.fromkeys(hit.chunk.source for hit in hits))
        ranks = [i for i, source in enumerate(ranked, 1) if source in relevant]
        dcg = sum(1 / log2(rank + 1) for rank in ranks)
        ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        rows.append({"query": case["query"], "sources": ranked,
                     "recall": len(ranks) / len(relevant),
                     "mrr": 1 / ranks[0] if ranks else 0.0, "ndcg": dcg / ideal})
    return {"k": k, "cases": rows, **{name: sum(r[name] for r in rows) / len(rows)
            for name in ("recall", "mrr", "ndcg")}}
