Rules:
- Use only source ids as table names and schema columns.
- For multi-table SQL, prefer `hints.join_candidates` when they are present; if no candidate fits a question, choose a join only from schema/value evidence.
- Treat `hints.join_candidates` and `hints.join_paths` as a join graph. If the requested columns live in tables that are connected only through an intermediate table, include the bridge table and all required candidate joins.
- Do not join two different entity id columns merely because their numeric values overlap; use same-name/key candidates or an explicit bridge-table path.
- Use `hints.question_value_matches` to ground WHERE literals to the table and column where the mentioned value occurs; these matches identify filters, not final answers.
- Use `hints.question_column_matches` to prefer columns explicitly mentioned by the question.
- In batch mode, `hints.question_value_matches_by_answer` and `hints.question_column_matches_by_answer` attach these matches to a specific answer name; apply an entry only to the matching output query.
- Double-quote tables, columns, and answer aliases.
- Each SELECT item must have a unique alias. If the requested answer is one string/list item but uses multiple fields (for example first and last name), concatenate them into one expression aliased as the matching answer name instead of returning separate answer columns.
- Avoid scalar subqueries in the SELECT list; write a flat SELECT with explicit FROM/JOIN clauses, or use derived tables in FROM when a subquery is necessary.
- Avoid `INTERSECT` and `EXCEPT`; express set intersections with joins, `EXISTS`,
  or `GROUP BY` with `HAVING COUNT(DISTINCT column)`.
- Return one item per `questions[i]`; order and length must match.
- Each item is a JSON object with exactly one key, `sql`, whose value is a
  complete SQLite SELECT statement.
- Compute deterministic totals, averages, differences, percentages, and ranks in SQL.
- For highest/lowest ties, return all tied rows; do not use LIMIT 1 unless one answer is guaranteed.
- Exclude summary rows such as total, all, overall, or all ages unless explicitly requested.
- Parenthesize mixed AND/OR filters.
- Parse years/dates from text cells or date-like column names before comparing them.
- No comments, markdown, or text outside the JSON array.
