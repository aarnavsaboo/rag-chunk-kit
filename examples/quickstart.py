from pathlib import Path
from rag_chunk_kit import SearchIndex, build_context, evaluate, ingest

here = Path(__file__).parent
index = SearchIndex.build(ingest(here / "corpus", max_chars=500, overlap=50))
hits = index.search("How does overlap affect retrieval?", k=3)
print(build_context(hits, max_chars=1400).text)
print("\nSources:", [hit.chunk.source for hit in hits])
