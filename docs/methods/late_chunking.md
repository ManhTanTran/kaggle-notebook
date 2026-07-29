# Late chunking

Late chunking creates vectors only after full-document contextual encoding. It
mean-pools states mapped to every chunk span, excludes padding/special tokens,
and normalizes on request. Long-document handling is explicit: `error` or
`truncate`; no silent truncation occurs.
