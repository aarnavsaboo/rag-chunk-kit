# Retrieval design notes

## Keep chunking inspectable

Every chunk has a source, heading path, stable ID and start/end offsets into the supplied string. IDs include source and span as well as text: identical passages in different documents remain distinguishable. Newline conversion by a file reader happens before offsets are computed. Whitespace is trimmed from emitted spans; non-heading content is otherwise retained.

The segmenter recognizes ATX Markdown headings and backtick/tilde fences. It is intentionally not a complete Markdown parser. Tables, lists and exceptionally long code blocks may be split when enforcing the character budget.

## Why two scores are not added directly

BM25 and cosine scores have different scales. The hybrid path first ranks positive-scoring candidates independently and combines reciprocal ranks with a constant of 60. `alpha` sets the dense channel's weight. Ties are deterministic in ingestion order. Nonpositive cosine candidates are excluded, which is a deliberate baseline policy rather than a universal retrieval recommendation.

Vectors are normalized when building/loading an index. The model identifier and dimensions must match at query time. Identifiers are not cryptographic model provenance: pin a model revision when running comparisons. Local JSON persistence favors transparency over index size and load time.

## Evaluation contract

Input cases contain `query` and a nonempty list of `relevant_sources`. Retrieve the top k chunks, then deduplicate their sources in rank order. Recall, reciprocal rank and binary nDCG are computed on that source list and macro-averaged. Multiple hits from the same document cannot inflate the number of relevant sources. Duplicate-heavy top-k results may leave fewer than k distinct sources. These metrics describe retrieval, not answer quality.

Use representative queries and held-out labels. The bundled three-query corpus is a runnable fixture only. There are no published performance claims.

## Integration boundary

`build_context` produces numbered excerpts and source records. Pass that context to a generation component of your choice. A citation identifies the supplied text span; it does not guarantee that a generated answer is supported by that span.

References: [Sentence Transformers encoding API](https://sbert.net/docs/package_reference/sentence_transformer/model.html), [Python data classes](https://docs.python.org/3/library/dataclasses.html).
