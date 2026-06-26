Profile: analyze.

{% if task_dsl.get("hints") -%}
Hints: {{ task_dsl.get("hints") | tojson }}
{% endif -%}

Return exactly one JSON object. No ids/wrapper.

Choose one mutually exclusive action form:

Evidence SQL object:
`op` must be `"sql"` and `q` must be one complete SQLite SELECT using real
source ids and columns.

Evidence analyze object:
`op` must be `"analyze"`, `kind` must be `"statistical"` or `"correlation"`,
and `seed` must be one complete SQLite SELECT using real source ids and columns.

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Rules:
- Return one action only. Do not return `acts`, `actions`, `sqls`, arrays, or
  multiple SQL statements.
- Do not output placeholders, angle-bracket text, or ellipsis characters; output
  executable SQL only.
- SQL actions retrieve evidence only; they do not encode final answers.
- If several evidence steps seem useful, fold them into one SELECT using joins,
  CTEs, derived tables, aggregates, or `EXISTS`.
- Prefer SQL for exact deterministic work: totals, averages, differences, percentages, ranks, and ties.
- Use analyze only when the requested statistic cannot be expressed directly in SQL.
- Exclude summary rows such as total, totals, all, overall, subtotal, rank total,
  or all ages unless the question explicitly asks for the overall total.
- For `who`/`which` questions, project the requested entity column (person,
  team, country, title, event, location, etc.), not the column used only for
  filtering, ranking, or comparison.
- For "which X has the most/least value" questions, filter out summary rows
  before ranking and return the X value, not the numeric max/min value.
- Project only requested answer fields. Keep helper values used for ranking,
  sorting, or filtering inside `ORDER BY`, CTEs, or derived tables unless the
  question explicitly asks to output them.
- For inclusive ranges such as `2000-2007`, `from 2000 to 2007`, or `between
  2000 and 2007`, include every row in the range; do not use `IN (2000, 2007)`
  unless only the endpoints are requested.
- For change/difference questions, preserve direction: `from A to B` is
  value_at_B - value_at_A; `increase compared to previous` is current -
  previous; `decrease compared to previous` is the most negative current -
  previous change.
- For average annual increase/decrease from year A to year B, compute
  `(value_at_B - value_at_A) / (B - A)` unless the question asks for the
  average of all values.
- For top/bottom-N aggregate questions, first select the top/bottom N rows in a
  CTE or derived table and aggregate that subset; do not aggregate before
  `ORDER BY ... LIMIT N`.
- For highest/lowest/ranking questions, preserve ties unless exactly one row is
  requested. Use a max/min CTE or `DENSE_RANK` and return all tied answer values.
- For questions asking how many times the top entity occurs, return the grouped
  count/sum for that entity, not `COUNT(*)` of the single top grouped row.
- For one-row wide tables with multiple year/period columns, compute averages
  across columns explicitly by dividing by the number of columns; do not use
  `AVG(col1 + col2 + ...)` over one row.
- For period-label ranges like `1950-1955 to 1975-1980`, include all ordered
  periods between the endpoints when a total/average over the range is asked.
- For comparisons between two named entities, compute separate scalar values
  for each entity in CTEs/self-joins/conditional aggregates before subtracting
  or comparing; do not compare a grouped row to itself.
- When a column stores quantities such as population, seats, counts, mintage, or
  number of locomotives, "total number" usually means `SUM(quantity_column)`,
  not `COUNT(*)`.
- When casting numeric text, remove commas, percent signs, and units, but never
  remove the decimal point itself.
- For yes/no questions, answer yes/no (or boolean only when the declared answer
  type is boolean). For "A or B" comparison questions, return the winning option
  text (A or B), not true/false.
- For "came first", "earlier", "last", "before", and "after", compare actual
  dates/row order, not raw strings.
- Parenthesize mixed AND/OR filters.
- For text predicates, prefer case-insensitive matching with `LOWER(column) LIKE`
  when the question wording may differ in case or spacing from cell text.
- Use terminal answer only when no table evidence is needed.
