Profile: query.
Return exactly one JSON object with one key, `sql`; its value must be an
executable SQLite SELECT over the provided source ids and columns.
Never return `acts`, `actions`, `sqls`, arrays, or multiple SQL statements.
Never output placeholders, angle-bracket text, or ellipsis characters.
