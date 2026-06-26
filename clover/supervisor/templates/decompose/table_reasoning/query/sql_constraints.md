SQL rules:
- The system processes exactly one question per request.
- Return exactly one SQL object for that question. Do not return arrays,
  `questions`, `answers`, `acts`, `actions`, `sqls`, or multiple commands.
- Never output placeholder SQL, angle-bracket text, or ellipsis characters. The
  SQL must be complete SQLite-compatible syntax using real source ids and real
  schema columns from the Task DSL.
- Use only source ids as table names and schema columns.
- For multi-table SQL, prefer `hints.join_candidates` when they are present; if no candidate fits the question, choose a join only from schema/value evidence.
- Treat `hints.join_candidates` and `hints.join_paths` as a join graph. If the requested columns live in tables that are connected only through an intermediate table, include the bridge table and all required candidate joins.
- Do not join two different entity id columns merely because their numeric values overlap; use same-name/key candidates or an explicit bridge-table path.
- Use `hints.question_value_matches` to ground WHERE literals to the table and column where the mentioned value occurs; these matches identify filters, not final answers.
- Use `hints.question_column_matches` to prefer columns explicitly mentioned by the question.
- Double-quote tables, columns, and answer aliases.
- Each SELECT item must have a unique alias. If the requested answer is one string/list item but uses multiple fields (for example first and last name), concatenate them into one expression aliased as the answer name instead of returning separate answer columns.
- If the question asks for multiple answer fields (for example "which city ...
  and what is the difference"), return all requested fields. For string answer
  types, prefer one concatenated answer expression such as
  `city || ', ' || difference` aliased as the answer name.
- Avoid scalar subqueries in the SELECT list; write a flat SELECT with explicit FROM/JOIN clauses, or use derived tables in FROM when a subquery is necessary.
- Avoid `INTERSECT` and `EXCEPT`; express set intersections with joins, `EXISTS`,
  or `GROUP BY` with `HAVING COUNT(DISTINCT column)`.
- The SQL is one read-only SELECT. If several evidence steps seem useful, fold
  them into one SELECT using joins, CTEs, derived tables, aggregates, or
  `EXISTS`; do not ask for multiple SQL statements.
- The final answer expression must be aliased as the requested answer name. For
  list or multi-row answers, return rows from that single SELECT.
- No markdown, comments, or extra text.
- If answer type is `string`, `number`, `boolean`, or `entity`, return a single row (`LIMIT 1` or aggregate).
- Prefer `LOWER(column) LIKE` for text filters when case or punctuation may differ.
- For `who`/`which` questions, SELECT the requested entity column, not the
  evidence column used for filtering, ranking, or sorting.
- Project only the requested answer fields. Keep ranking/sorting/numeric
  evidence columns in `ORDER BY`, CTEs, or derived tables unless the question
  explicitly asks to output those evidence values.
- Exclude rows whose entity/label column is total, totals, overall, subtotal, or
  all unless the question explicitly asks for a total/overall row.
- When ordering dates, parse/normalize dates where possible; do not rely on
  lexicographic text ordering for month names.
{% if task_dsl.get("hints", {}).get("benchmark") == "tablebench" -%}
- TableBench arithmetic rules:
- For ranges like `2000-2007`, `from 2000 to 2007`, or `between 2000 and 2007`,
  include every row in the inclusive range; do not filter only the two endpoint
  years unless the question explicitly asks for endpoint values.
- For change/difference questions, preserve direction:
  `from A to B` means value_at_B - value_at_A; `increase compared to previous`
  means current - previous; `decrease compared to previous` means the most
  negative current - previous change (or equivalently the largest
  previous - current drop).
- For average annual increase/decrease from year A to year B, compute
  `(value_at_B - value_at_A) / (B - A)` unless the question explicitly asks for
  the average of all yearly values.
- For top/bottom-N aggregate questions, first select the top/bottom N rows in a
  CTE or derived table, then aggregate that subset. Do not write
  `SUM(...) ... ORDER BY ... LIMIT N`, because the aggregate would be computed
  before the limit.
- For highest/lowest/ranking questions, preserve ties unless the question asks
  for exactly one row. Use a max/min CTE or `DENSE_RANK` and return all rows
  tied at the requested rank.
- For questions that ask how many times the top entity occurs, return that top
  entity's aggregate count/sum itself; do not `COUNT(*)` around a derived table
  that already has one grouped row.
- If the table has one row with multiple year/period columns, compute averages
  or changes across those columns explicitly, e.g. `(col1 + col2 + col3) / 3.0`;
  `AVG(col1 + col2 + col3)` over one row is a sum, not an average.
- For period-label ranges such as `1950-1955 to 1975-1980`, include every
  ordered period between the two labels when the question asks a total/average
  over the range, not just the endpoints.
- For comparisons between two named entities, compute each entity's value in a
  separate CTE/self-join/conditional aggregate, then subtract or compare those
  two scalar values. Do not group rows and compare a value to itself.
- When a column stores quantities such as population, seats, counts, mintage, or
  number of locomotives, "total number" usually means `SUM(quantity_column)`,
  not `COUNT(*)` rows.
- When casting numeric text, remove commas, percent signs, and units, but never
  remove the decimal point itself.
{% endif -%}
