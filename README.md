# RAG Chunk Kit

A small document chunking CLI by **Aarnav Saboo** for experiments with retrieval-augmented generation.

It turns Markdown or plain text into predictable JSONL while retaining heading context and source metadata.

```bash
pip install -e .
rag-chunk notes.md --max-chars 1200 -o chunks.jsonl
```

Example output:

```json
{"source":"notes.md","index":0,"heading":"Architecture","text":"..."}
```

The goal is intentionally simple chunking that is easy to inspect before moving to more sophisticated retrieval strategies.

Built by **Aarnav Saboo**.

https://aarnavsaboo.github.io
