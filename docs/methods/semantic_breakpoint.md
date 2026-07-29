# Semantic breakpoint

Sentence segments are embedded through an injected embedder. Cosine distances of
adjacent segments are thresholded by percentile, absolute, standard-deviation, or
a custom callable. Small groups may be merged and a declared maximum can split
only at segment boundaries.
