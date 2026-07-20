# Context guide

The `world` object in the file payload contains everything you need:

- `source_sql`: the global plan's SQL from which this node was compiled. Read it to understand the intended transformation.
- `inputs`: one entry per solve argument. Each entry has:
  - `rows`: row count.
  - `cols`: column names.
  - `sample_values`: top distinct values per column (up to 3). Use them to detect text formats (e.g. "$1.2M", "Name (CODE)", "2023-01-15") and pick the right parsing strategy.
- `output_contract`: required shape of the result (columns, single_row, non_empty).
- `few_shot_hint`: one-line pattern hint tailored to this node's op. Follow it when present.
- `diag`: present only after a fast-path failure. Contains `head` records, `values`, and `ranges` for referenced columns.

Before writing `solve`:
1. Read `source_sql` to understand what this node should compute.
2. Inspect `inputs[*].sample_values` to learn the data format.
3. Check `output_contract` to know the required result shape.
4. If `diag` is present, use its `values` to relax matches.
