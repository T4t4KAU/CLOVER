SQL rules:
- Use only source ids as table names and schema columns.
- Double-quote tables, columns, and answer aliases.
- Each `sql` is one read-only SELECT.
- No markdown, comments, or extra text.
- If answer type is `string`, `number`, `boolean`, or `entity`, return a single row (`LIMIT 1` or aggregate).
