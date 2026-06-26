{% if task_dsl.get("profile") == "analyze" -%}
Task DSL:

{{TASK_DSL}}

Plan JSON:
{%- else -%}
Task DSL:

{{TASK_DSL}}

Return one JSON object with exactly one key, `sql`. Its value must be one
complete SQLite SELECT statement that aliases the final answer expression as
`{{ answer.get("name", "answer") }}`.

Never return a JSON array, `acts`, `actions`, `sqls`, `questions`, or more than
one SQL statement. Never output placeholders, angle-bracket text, or ellipsis
characters.
{%- endif %}
