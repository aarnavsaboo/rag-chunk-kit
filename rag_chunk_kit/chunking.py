"""Offset-preserving Markdown segmentation for retrieval experiments."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import re


@dataclass(frozen=True)
class Chunk:
    index: int
    heading: str | None
    text: str
    source: str = "document"
    start: int = 0
    end: int = 0
    id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 120,
                   source: str = "document") -> list[Chunk]:
    """Split by Markdown sections, then bounded windows; offsets are code points.

    Heading markers inside fenced code are not treated as section boundaries.
    A character budget is deliberately not described as a token budget.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_chars < 1 or not 0 <= overlap < max_chars:
        raise ValueError("require max_chars > 0 and 0 <= overlap < max_chars")
    sections: list[tuple[int, int, str | None]] = []
    headings: list[tuple[int, str]] = []
    section_start = offset = 0
    heading = None
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not tail.strip():
                fence = None
        elif fence is None:
            match = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                sections.append((section_start, offset, heading))
                level, title = len(match[1]), match[2]
                headings = [(n, value) for n, value in headings if n < level]
                headings.append((level, title))
                heading = " / ".join(value for _, value in headings)
                section_start = offset + len(line)
        offset += len(line)
    sections.append((section_start, len(text), heading))
    chunks = []
    for left, right, heading in sections:
        while left < right:
            stop = min(left + max_chars, right)
            if stop < right:
                # Prefer a natural boundary in the second half of the window.
                matches = list(re.finditer(r"\n\s*\n|[.!?]\s+|\s+", text[left:stop]))
                useful = [m.end() for m in matches if m.end() > max_chars // 2]
                if useful:
                    stop = left + useful[-1]
            start, end = left, stop
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                body = text[start:end]
                key = f"{source}\0{start}\0{end}\0{body}".encode()
                chunks.append(Chunk(len(chunks), heading, body, source, start,
                                    end, sha256(key).hexdigest()[:20]))
            if stop == right:
                break
            left = max(left + 1, stop - overlap)
    return chunks
