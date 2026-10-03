# Retrieval experiments

Chunking and retrieval changes should be measured together with the corpus they are intended to serve.

The experiment runner expands a small JSON manifest into concrete configurations. Each configuration rebuilds the index and evaluates the same labelled queries.

For useful comparisons, avoid changing chunking, embedding model and top-k simultaneously when the goal is to understand causality. Large grids are useful for exploration; smaller paired sweeps are better for explaining a result.

Local query expansion is deliberately separate from the retrieval runner because generated rewrites add another model-dependent variable. Store generated queries before comparing them.
