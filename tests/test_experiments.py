import unittest

from rag_chunk_kit.experiments import expand


class Tests(unittest.TestCase):
    def test_expand(self):
        rows=expand({
            "name":"x",
            "corpus":"c",
            "cases":"q",
            "chunk_chars":[300,600],
            "overlap":[0,50],
            "top_k":[5],
            "embedding_models":[None],
        })
        self.assertEqual(len(rows),4)
        self.assertEqual(len({x.id for x in rows}),4)


if __name__=="__main__":
    unittest.main()
