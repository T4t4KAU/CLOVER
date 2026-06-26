SQL rules:
- Use only source ids as table names and schema columns.
- Double-quote tables, columns, and answer aliases.
- Each `sql` is one read-only SELECT.
- No markdown, comments, or extra text.
- If answer type is `string`, `number`, `boolean`, or `entity`, return a single row (`LIMIT 1` or aggregate).
- Prefer `LOWER(...) LIKE` for text filters when case or punctuation may differ.
- For `who`/`which` questions, SELECT the requested entity column, not the
  evidence column used for filtering, ranking, or sorting.
- Exclude rows whose entity/label column is total, totals, overall, subtotal, or
  all unless the question explicitly asks for a total/overall row.
- When ordering dates, parse/normalize dates where possible; do not rely on
  lexicographic text ordering for month names.
