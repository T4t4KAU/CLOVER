SQL rules:
- Use only source ids as table names and schema columns.
- For multi-table SQL, prefer `hints.join_candidates` when they are present; if no candidate fits the question, choose a join only from schema/value evidence.
- Treat `hints.join_candidates` and `hints.join_paths` as a join graph. If the requested columns live in tables that are connected only through an intermediate table, include the bridge table and all required candidate joins.
- Do not join two different entity id columns merely because their numeric values overlap; use same-name/key candidates or an explicit bridge-table path.
- Use `hints.question_value_matches` to ground WHERE literals to the table and column where the mentioned value occurs; these matches identify filters, not final answers.
- Use `hints.question_column_matches` to prefer columns explicitly mentioned by the question.
- Double-quote tables, columns, and answer aliases.
- Each SELECT item must have a unique alias. If the requested answer is one string/list item but uses multiple fields (for example first and last name), concatenate them into one expression aliased as the answer name instead of returning separate answer columns.
- Avoid scalar subqueries in the SELECT list; write a flat SELECT with explicit FROM/JOIN clauses, or use derived tables in FROM when a subquery is necessary.
- Avoid `INTERSECT` and `EXCEPT`; express set intersections with joins, `EXISTS`, or `GROUP BY ... HAVING COUNT(DISTINCT ...)`.
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
