# RAG Chunk Kit

**A local document-to-context pipeline for RAG experiments.**

Chunking is only useful if you can see what it does to retrieval. This project connects Markdown ingestion, source-preserving chunks, BM25, optional dense embeddings, reciprocal-rank fusion and labelled retrieval evaluation in one inspectable pipeline.

Maintained by **Aarnav Saboo**. Python 3.10+, MIT. The lexical pipeline uses only the standard library.

## Run a complete experiment

```bash
python -m pip install -e .
rag-kit index examples/corpus -o local-index.json --max-chars 500 --overlap 50
rag-kit search local-index.json "How does chunk overlap work?" -k 3
rag-kit context local-index.json "How does chunk overlap work?" --max-chars 1400
rag-kit evaluate local-index.json examples/queries.json -k 3
```

No API key or downloaded model is needed for these commands. The sample corpus and labels are deliberately tiny fixtures, not a benchmark of real-world retrieval quality.

## The pipeline

```text
Markdown / text
  -> section detection + bounded overlapping windows
  -> chunks with stable IDs, headings and character offsets
  -> BM25 index (+ optional normalized embedding vectors)
  -> lexical / cosine ranks -> reciprocal-rank fusion
  -> context budget + numbered source citations
  -> source-level Recall@k, MRR@k and nDCG@k
```

## Use from Python

```python
from rag_chunk_kit import SearchIndex, build_context, ingest

chunks = ingest("examples/corpus", max_chars=500, overlap=50)
index = SearchIndex.build(chunks)
hits = index.search("embedding model dimensions", k=3)
context = build_context(hits, max_chars=1600)
print(context.text)
print(context.citations)
```

`search(..., source="retrieval.md")` applies an exact source filter. Each result keeps its lexical score and optional cosine score separate from its final fusion score. That final score is a rank signal, not a confidence or probability.

## Optional dense retrieval

```bash
python -m pip install -e '.[embeddings]'
```

```python
from rag_chunk_kit import SentenceTransformerEmbedder, SearchIndex, ingest

# This explicitly loads/downloads the chosen model. Pin a revision for reproducibility.
encoder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
index = SearchIndex.build(ingest("examples/corpus"), encoder)
index.save("local-index.json")
hits = index.search("how to compare text vectors", embedder=encoder, alpha=0.5)
```

The CLI accepts `--embedding-model` for `index`, `search`, `context` and `evaluate`. Use the same model for indexing and queries. For a pinned model revision, use the Python API. Querying a saved dense index without an encoder intentionally uses BM25 only.

## What is implemented

- ATX heading hierarchies, fenced-code awareness and strict character limits.
- Overlapping windows with exact Python string offsets and stable chunk identifiers.
- Recursive `.md` and `.txt` ingestion with relative source paths.
- BM25 ranking and an optional Sentence Transformers embedding adapter.
- Cosine ranking and weighted reciprocal-rank fusion, with model/dimension checks.
- JSON index persistence, source filters and deduplicated context citations.
- Labelled, source-level retrieval metrics and an offline sample corpus.

## Trade-offs

This is an in-memory reference implementation for small collections, not a distributed vector database. Ranking scans the corpus. Embeddings are stored in the JSON index; no ANN index or reranker is included. The tokenizer is Unicode word matching, not a language-specific morphological analyzer. Chunk budgets count characters, **not model tokens**. Long code blocks can be split across windows. Heading metadata is retained separately from passage text.

The package builds retrieval context; it does not generate an answer or claim to verify factual correctness. Dense retrieval depends on your chosen model. Tests use a deterministic toy encoder rather than downloaded model weights.

## Development

```bash
python -m unittest discover -s tests -v
python examples/quickstart.py
```

See [architecture and evaluation notes](docs/design.md). The earlier `rag-chunk file.md -o chunks.jsonl` entry point remains available.
