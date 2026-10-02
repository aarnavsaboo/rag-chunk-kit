import argparse
import json
import sys
from pathlib import Path
from . import SearchIndex, SentenceTransformerEmbedder, build_context, evaluate, ingest


def main():
    parser = argparse.ArgumentParser(description="Local RAG ingestion and retrieval workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("chunk", "index"):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("-o", "--output", required=True)
        p.add_argument("--max-chars", type=int, default=1200)
        p.add_argument("--overlap", type=int, default=120)
        if name == "index":
            p.add_argument("--embedding-model")
    for name in ("search", "context", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("index")
        p.add_argument("query" if name != "evaluate" else "cases")
        p.add_argument("-k", type=int, default=5)
        p.add_argument("--embedding-model")
        if name != "evaluate":
            p.add_argument("--source")
        if name == "context":
            p.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()
    try:
        if args.command in {"chunk", "index"}:
            chunks = ingest(args.input, args.max_chars, args.overlap)
            if args.command == "index":
                embedder = SentenceTransformerEmbedder(args.embedding_model) if args.embedding_model else None
                SearchIndex.build(chunks, embedder).save(args.output)
            else:
                Path(args.output).write_text("".join(json.dumps(c.as_dict(), ensure_ascii=False)
                                            + "\n" for c in chunks), encoding="utf-8")
            print(json.dumps({"chunks": len(chunks), "output": args.output}))
            return 0
        index = SearchIndex.load(args.index)
        embedder = SentenceTransformerEmbedder(args.embedding_model) if args.embedding_model else None
        if args.command == "evaluate":
            result = evaluate(index, json.loads(Path(args.cases).read_text(encoding="utf-8")), args.k, embedder)
        else:
            hits = index.search(args.query, args.k, source=args.source, embedder=embedder)
            result = (build_context(hits, args.max_chars).as_dict() if args.command == "context"
                      else [hit.as_dict() for hit in hits])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"rag-kit: {exc}", file=sys.stderr)
        return 2


def legacy_main():
    # Preserve the original one-file chunking command.
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()
    chunks = ingest(args.input, args.max_chars, min(120, max(0, args.max_chars - 1)))
    Path(args.output).write_text("".join(json.dumps(c.as_dict()) + "\n" for c in chunks))


if __name__ == "__main__":
    raise SystemExit(main())
