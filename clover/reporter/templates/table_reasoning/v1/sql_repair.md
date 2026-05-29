The current SQL failed locally.

Your task now is SQL repair, not final answer generation.

Use the original user task, table schema, current SQL, and local error in the report payload to generate one corrected SQL statement for the next local round.

Rules:

- Set retry to true.
- Set answer to null.
- Return exactly this JSON shape:

{
  "answer": null,
  "retry": true,
  "new_sql": {
    "sql": "SELECT ..."
  }
}

- Set new_sql to an object with exactly this shape:

{
  "sql": "SELECT ..."
}

- When writing new_sql.sql, reuse the SQL generation constraints already given to Commander earlier in this conversation.

Report payload:

{{REPORT_PAYLOAD}}

Reporter JSON:
