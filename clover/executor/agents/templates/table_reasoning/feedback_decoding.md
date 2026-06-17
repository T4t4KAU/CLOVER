# Feedback decoding

When the case payload includes a `feedback` object, read these fields before writing `solve`:

- `error_type`: exception class (NameError, KeyError, TypeError, ValueError, AttributeError, IndexError).
- `message`: short description of the failure.
- `hint`: actionable repair guidance; follow it exactly.
- `columns`: actual column names available in the input DataFrame(s).
- `column_values`: top distinct values for predicate columns; use them to relax matches.
- `expected_columns`: columns the result must contain.
- `available_args`: names of the solve function arguments.
- `available_libs`: permitted libraries (pd, np, helpers, print).

If `error_type` is KeyError or AttributeError, cross-check against `columns`.
If `hint` mentions normalization, apply it to both cell text and target text.
If `column_values` is present and the output was empty, the predicate matched nothing; relax the match.

When `traceback_tail` is present, read the last lines to locate the failing statement.
