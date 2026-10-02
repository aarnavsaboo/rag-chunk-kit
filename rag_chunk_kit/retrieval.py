"""BM25, optional dense retrieval, and reciprocal-rank fusion."""
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, log, sqrt
from pathlib import Path
from typing import Protocol, Sequence
import json
import re
from .chunking import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class Embedder(Protocol):
    id: str

    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEmbedder:
    """Optional dependency; model loading happens only when explicitly created."""
    def __init__(self, model: str, revision: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.id = f"sentence-transformers:{model}@{revision or 'default'}"
        self.model = SentenceTransformer(model, revision=revision)

    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def normalize(vector: Sequence[float]) -> list[float]:
    values = [float(x) for x in vector]
    if not values or not all(isfinite(x) for x in values):
        raise ValueError("embedding must be a nonempty finite vector")
    norm = sqrt(sum(x * x for x in values))
    if not isfinite(norm) or norm == 0:
        raise ValueError("embedding norm must be positive and finite")
    return [x / norm for x in values]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    lexical_score: float
    dense_score: float | None

    def as_dict(self) -> dict:
        return asdict(self)


class SearchIndex:
    def __init__(self, chunks: list[Chunk], vectors=None, embedding_id=None):
        self.chunks = list(chunks)
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        self.counts = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(1, len(chunks)) or 1.0
        df = Counter(word for counts in self.counts for word in counts)
        self.idf = {w: log(1 + (len(chunks) - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}
        self.vectors = None if vectors is None else [normalize(v) for v in vectors]
        self.embedding_id = embedding_id
        if self.vectors is not None:
            dimensions = {len(v) for v in self.vectors}
            if len(self.vectors) != len(chunks) or len(dimensions) > 1 or not embedding_id:
                raise ValueError("vectors must align with chunks and declare their model")

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None):
        vectors = embedder.encode([c.text for c in chunks]) if embedder and chunks else None
        return cls(chunks, vectors, embedder.id if embedder else None)

    def search(self, query: str, k: int = 5, *, source: str | None = None,
               embedder: Embedder | None = None, alpha: float = 0.5) -> list[Hit]:
        """Fuse BM25 and cosine ranks; alpha is the dense contribution, not similarity."""
        if k < 1 or not 0 <= alpha <= 1:
            raise ValueError("require k > 0 and alpha between 0 and 1")
        if not query.strip() or not self.chunks:
            return []
        eligible = [i for i, c in enumerate(self.chunks) if source is None or c.source == source]
        words = set(tokenize(query))
        lexical = {}
        for i in eligible:
            counts = self.counts[i]
            lexical[i] = sum(self.idf.get(w, 0) * counts[w] * 2.5 /
                             (counts[w] + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg_length))
                             for w in words if counts[w])
        dense = {}
        if embedder is not None:
            if self.vectors is None or embedder.id != self.embedding_id:
                raise ValueError("query embedder must match the indexed model")
            encoded = list(embedder.encode([query]))
            if len(encoded) != 1:
                raise ValueError("embedder must return one vector per input")
            q = normalize(encoded[0])
            if self.vectors and len(q) != len(self.vectors[0]):
                raise ValueError("query embedding dimension mismatch")
            dense = {i: sum(a * b for a, b in zip(q, self.vectors[i])) for i in eligible}
        scores: dict[int, float] = {}
        sources = [(lexical, 1 - alpha), (dense, alpha)] if dense else [(lexical, 1.0)]
        for raw, weight in sources:
            if not weight:
                continue
            ranked = sorted((i for i in raw if raw[i] > 0), key=lambda i: (-raw[i], i))
            for rank, i in enumerate(ranked, 1):
                scores[i] = scores.get(i, 0) + weight / (60 + rank)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [Hit(self.chunks[i], scores[i], lexical[i], dense.get(i)) for i in ordered]

    def save(self, path: str | Path) -> None:
        data = {"version": 1, "chunks": [c.as_dict() for c in self.chunks],
                "vectors": self.vectors, "embedding_id": self.embedding_id}
        target = Path(path)
        target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("unsupported index version")
        return cls([Chunk(**c) for c in data["chunks"]], data["vectors"], data["embedding_id"])
