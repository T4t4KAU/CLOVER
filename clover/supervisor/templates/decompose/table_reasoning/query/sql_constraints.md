SQL rules:
- Use only source ids as table names and schema columns.
- For multi-table SQL, prefer `hints.join_candidates` when they are present; if no candidate fits the question, choose a join only from schema/value evidence.
- Treat `hints.join_candidates` and `hints.join_paths` as a join graph. If the requested columns live in tables that are connected only through an intermediate table, include the bridge table and all required candidate joins.
- Do not join two different entity id columns merely because their numeric values overlap; use same-name/key candidates or an explicit bridge-table path.
- Double-quote tables, columns, and answer aliases.
- Each `sql` is one read-only SELECT.
- No markdown, comments, or extra text.
- If answer type is `string`, `number`, `boolean`, or `entity`, return a single row (`LIMIT 1` or aggregate).
