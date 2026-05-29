SQL generation constraints:

- Use only the table sources provided in the task DSL.
- For each table source, use its `id` field as the SQL table name.
- Quote table identifiers with double quotes.
- Use only the columns provided in the table schema.
- Quote column identifiers with double quotes when generating SQL.
- Generate exactly one SQL statement for each question in `questions`.
- Return exactly one JSON array of SQL strings.
- The array length must equal the length of `questions`.
- The SQL string at index i must answer `questions[i]`.
- The SQL string at index i must return the answer using the alias `answers[i].name`.
- Quote the answer alias with double quotes, for example `AS "answer_1"`.
- Do not restart answer numbering. Use the answer names exactly as given.
- Each SQL statement should directly answer its corresponding question whenever possible.
- Do not include comments inside SQL statements.
- Do not include any text before or after the JSON array.
