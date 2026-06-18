Return only JSON: {"action":"rewrite_predicate","predicate":"WHERE ..."}.
The predicate must be a valid SQL WHERE clause fragment.
No markdown. No prose. No code outside the JSON.

You are repairing a Filter that returned 0 rows. The SQL literals do not match
the actual column values. Rewrite the WHERE predicate so the literals match the
actual values exactly.

Rules:
- Compare sql_lit against actual in each root to find the format difference.
- Rewrite only the literals to match the actual values exactly.
- Keep the same columns, operators, and structure.
- If a root has mismatch "quoting", the actual values have extra quote chars
  that the SQL literal is missing. Wrap the literal to match.
- If a root has mismatch "format", normalize case/spacing to match actual.
- If candidates are present, a different column may contain the target value;
  switch the predicate column only if the current column clearly cannot match.

Case:
{{ CASE_JSON }}
