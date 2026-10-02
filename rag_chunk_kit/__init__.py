"""Inspectable retrieval pipelines for local document collections."""
from .chunking import Chunk, chunk_markdown
from .retrieval import Hit, SearchIndex, SentenceTransformerEmbedder
from .pipeline import Context, build_context, evaluate, ingest

__all__ = ["Chunk", "chunk_markdown", "Hit", "SearchIndex", "SentenceTransformerEmbedder",
           "Context", "build_context", "evaluate", "ingest"]
