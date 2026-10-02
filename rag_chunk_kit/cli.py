import argparse
import json
from pathlib import Path

from . import chunk_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Chunk Markdown/text into JSONL for retrieval pipelines"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    source = Path(args.input)
    chunks = chunk_markdown(
        source.read_text(encoding="utf-8"),
        args.max_chars,
    )

    with open(args.output, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            record = {"source": source.name, **chunk.as_dict()}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
