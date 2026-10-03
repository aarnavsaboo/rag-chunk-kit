# RAG Chunk Kit

A local retrieval workbench for document ingestion, chunking, hybrid search and repeatable RAG experiments.

The package keeps retrieval mechanics inspectable: source-preserving chunks, BM25, optional local embeddings, reciprocal-rank fusion, evidence assembly, labelled retrieval metrics and experiment sweeps. It is deliberately smaller than a full RAG framework so individual retrieval decisions remain visible.

## Core pipeline

```text
Markdown / text
      |
      v
section-aware chunking
      |
      +--> stable IDs
      +--> source paths
      +--> headings
      +--> exact offsets
      |
      v
BM25 index + optional dense vectors
      |
      v
lexical / dense rankings
      |
      v
reciprocal-rank fusion
      |
      v
evidence assembly
      |
      v
Recall@k / MRR / nDCG
```

## Quick start

```bash
python -m pip install -e .

rag-kit index examples/corpus -o runs/index.json \
  --max-chars 600 \
  --overlap 80

rag-kit search runs/index.json \
  "How does reciprocal-rank fusion combine results?" \
  -k 5
```

Dense retrieval is optional:

```bash
python -m pip install -e '.[embeddings]'

rag-kit index examples/corpus \
  -o runs/dense.json \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

## Local query expansion experiments

The package includes an optional localhost model adapter for generating alternate search queries.

```python
from rag_chunk_kit.local_query import OllamaQueryExpander

expander = OllamaQueryExpander("qwen3:4b")
queries = expander.expand(
    "Why can lexical and dense retrieval complement each other?",
    count=3,
)
```

Generated rewrites are kept as experiment artifacts rather than silently replacing the original query.

## Retrieval experiment manifests

```json
{
  "name": "chunk-sweep",
  "corpus": "examples/corpus",
  "cases": "examples/queries.json",
  "chunk_chars": [300, 600, 1200],
  "overlap": [0, 60, 120],
  "top_k": [3, 5, 10],
  "embedding_models": [null]
}
```

Expand and run the matrix:

```bash
python -m rag_chunk_kit.experiment_cli plan configs/experiment.example.json > runs/plan.jsonl
python -m rag_chunk_kit.experiment_cli run runs/plan.jsonl --out runs/results.jsonl
python -m rag_chunk_kit.experiment_cli report runs/results.jsonl
```

Each run stores the exact configuration, aggregate retrieval metrics and per-query rankings. This makes chunk-size and overlap changes measurable rather than subjective.

## What is implemented

- ATX heading-aware Markdown chunking
- fenced-code awareness
- bounded overlapping windows
- exact character offsets
- stable chunk IDs
- recursive Markdown/text ingestion
- BM25 retrieval
- optional Sentence Transformers embeddings
- cosine ranking
- reciprocal-rank fusion
- persistent JSON indexes
- source filtering
- labelled retrieval evaluation
- local query expansion experiments
- config-driven chunk/retrieval sweeps

## What is intentionally not hidden

The project does not pretend every additional RAG stage is useful. Dense retrieval may lose to BM25 on a corpus. Query expansion may introduce worse search terms. Larger chunks may increase apparent recall while reducing passage specificity.

The experiment tooling keeps those outcomes visible.

## Repository layout

- `rag_chunk_kit/chunking.py` — document segmentation
- `rag_chunk_kit/retrieval.py` — lexical/dense retrieval and fusion
- `rag_chunk_kit/pipeline.py` — ingestion, evidence assembly and metrics
- `rag_chunk_kit/local_query.py` — optional localhost query expansion
- `rag_chunk_kit/experiments.py` — run specs and sweep execution
- `rag_chunk_kit/experiment_report.py` — grouped experiment summaries
- `docs/` — design and experiment notes
- `examples/` — tiny offline fixtures
- `tests/` — deterministic tests

Maintained by **Aarnav Saboo**.
