import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from rag_chunk_kit import SearchIndex, build_context, chunk_markdown, evaluate, ingest
from rag_chunk_kit.retrieval import normalize

class ToyEmbedder:
    id = "toy-v1"
    def encode(self, texts):
        return [[1 + t.lower().count("cat"), 1 + t.lower().count("dog")] for t in texts]

class ChunkTests(unittest.TestCase):
    def test_budget_and_offsets(self):
        text = "# Topic\n" + "A sentence with several words. " * 40
        chunks = chunk_markdown(text, 100, 20)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(len(c.text), 100)
            self.assertEqual(c.text, text[c.start:c.end])
            self.assertEqual(c.heading, "Topic")
    def test_heading_without_blank_line(self):
        chunks = chunk_markdown("# One\nAlpha\n## Two\nBeta", 100, 0)
        self.assertEqual([c.heading for c in chunks], ["One", "One / Two"])
        self.assertEqual([c.text for c in chunks], ["Alpha", "Beta"])
    def test_fences(self):
        text = "# A\n```python\n# not a heading\nx = 1\n```\nTail"
        self.assertEqual(len(chunk_markdown(text, 200, 0)), 1)
        self.assertIn("# not a heading", chunk_markdown(text, 200, 0)[0].text)
    def test_tilde_fences(self):
        chunks = chunk_markdown("~~~text\n# not heading\n~~~", 100, 0)
        self.assertIsNone(chunks[0].heading)
    def test_empty(self):
        self.assertEqual(chunk_markdown("   "), [])
    def test_unicode(self):
        text = "# हिंदी\nभाषा और डेटा " * 6
        for c in chunk_markdown(text, 40, 3):
            self.assertEqual(c.text, text[c.start:c.end])
    def test_invalid_budget(self):
        for size, overlap in [(0,0),(10,10),(10,-1)]:
            with self.assertRaises(ValueError): chunk_markdown("abc", size, overlap)
    def test_long_unbroken_input(self):
        chunks = chunk_markdown("x" * 203, 20, 0)
        self.assertEqual("".join(c.text for c in chunks), "x" * 203)
    def test_ids_are_stable_and_source_specific(self):
        a = chunk_markdown("content", 100, 0, "a")
        self.assertEqual(a, chunk_markdown("content", 100, 0, "a"))
        self.assertNotEqual(a[0].id, chunk_markdown("content", 100, 0, "b")[0].id)

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = (chunk_markdown("cat kitten animal", 100, 0, "cats.md") +
                       chunk_markdown("dog puppy animal", 100, 0, "dogs.md"))
    def test_bm25(self):
        self.assertEqual(SearchIndex(self.chunks).search("kitten")[0].chunk.source, "cats.md")
    def test_source_filter(self):
        self.assertEqual(SearchIndex(self.chunks).search("kitten", source="dogs.md"), [])
    def test_empty_query_and_corpus(self):
        self.assertEqual(SearchIndex(self.chunks).search(" "), [])
        self.assertEqual(SearchIndex([]).search("cat", embedder=ToyEmbedder()), [])
    def test_unknown_terms(self):
        self.assertEqual(SearchIndex(self.chunks).search("quantum"), [])
    def test_dense_and_hybrid(self):
        e = ToyEmbedder(); index = SearchIndex.build(self.chunks, e)
        self.assertEqual(index.search("cat cat", embedder=e)[0].chunk.source, "cats.md")
        self.assertIsNotNone(index.search("cat", embedder=e)[0].dense_score)
    def test_model_mismatch(self):
        e = ToyEmbedder(); index = SearchIndex.build(self.chunks, e); e.id = "v2"
        with self.assertRaises(ValueError): index.search("cat", embedder=e)
    def test_dimension_mismatch(self):
        e = ToyEmbedder(); index = SearchIndex.build(self.chunks, e)
        e.encode = lambda texts: [[1,2,3] for _ in texts]
        with self.assertRaises(ValueError): index.search("cat", embedder=e)
    def test_vectors_rejected(self):
        for vector in [[],[0,0],[float("nan")],[float("inf")]]:
            with self.assertRaises(ValueError): normalize(vector)
    def test_duplicate_ids(self):
        with self.assertRaises(ValueError): SearchIndex([self.chunks[0]] * 2)
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index = SearchIndex.build(self.chunks, ToyEmbedder()); index.save(path)
            other = SearchIndex.load(path)
            self.assertEqual(other.search("cat"), index.search("cat"))
            self.assertEqual(other.embedding_id, "toy-v1")
    def test_context_budget_and_citations(self):
        hits = SearchIndex(self.chunks).search("animal")
        context = build_context(hits + hits, 38)
        self.assertLessEqual(len(context.text), 38)
        self.assertEqual(len({c["chunk_id"] for c in context.citations}), len(context.citations))
    def test_evaluation(self):
        result = evaluate(SearchIndex(self.chunks), [{"query":"kitten", "relevant_sources":["cats.md"]}], 1)
        self.assertEqual((result["recall"],result["mrr"],result["ndcg"]), (1,1,1))
    def test_ingest_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"notes.md").write_text("# Models\nModel routing notes.")
            self.assertEqual(len(ingest(root)), 1)
            cmd=[sys.executable,"-m","rag_chunk_kit","index",str(root),"-o",str(root/"index.json")]
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            result=subprocess.run([sys.executable,"-m","rag_chunk_kit","search",str(root/"index.json"),"routing"],capture_output=True,text=True)
            self.assertEqual(json.loads(result.stdout)[0]["chunk"]["source"],"notes.md")

if __name__ == "__main__": unittest.main()
