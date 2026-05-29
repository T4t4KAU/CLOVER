SQL generation constraints:

- Use only the table sources provided in the task DSL.
- For each table source, use its `id` field as the SQL table name.
- Quote table identifiers with double quotes.
- Use only the columns provided in the table schema.
- Quote column identifiers with double quotes when generating SQL.
- Generate exactly one SQL statement.
- The SQL should directly answer the user question whenever possible.
- Return only the SQL statement.
- Do not include comments.
- Do not include any text before or after the SQL statement.
