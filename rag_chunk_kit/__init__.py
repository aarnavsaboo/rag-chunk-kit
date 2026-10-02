from dataclasses import dataclass, asdict
import re


@dataclass
class Chunk:
    index: int
    heading: str | None
    text: str

    def as_dict(self):
        return asdict(self)


def chunk_markdown(text: str, max_chars: int = 1200) -> list[Chunk]:
    heading = None
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush():
        nonlocal buffer
        body = "\n\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(len(chunks), heading, body))
        buffer = []

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue

        if block.startswith("#"):
            flush()
            heading = block.lstrip("#").strip()
            continue

        candidate = "\n\n".join(buffer + [block])
        if buffer and len(candidate) > max_chars:
            flush()

        buffer.append(block)

    flush()
    return chunks
