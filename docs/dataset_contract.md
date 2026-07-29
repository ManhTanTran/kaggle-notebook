# Dataset contract

Adapters return the existing `DatasetBundle` of `Document`, `Query`, and
`Evidence`. Evidence locators live in metadata and may include `granularity`,
`char_spans`, `sentence_ids`, `section_name`, `answer_spans`, and `raw_locator`.

Bundle metadata contains `dataset_name`, `split`, `language`,
`evaluation_profile`, `source`, and `adapter_version`.

Validation checks unique document/query identifiers, evidence identifiers within
each query, non-empty text, positive and consistent token counts, all references,
character-span bounds/text, and mapping quality. Diagnostics report counts,
unanswerable queries, duplicates, missing documents, mapping successes/failures,
rate, and explicit mapping errors. `fail_on_mapping_error` and
`minimum_mapping_rate` are dataset-configurable.
