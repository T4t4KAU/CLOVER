{% if task_dsl.get("profile") == "analyze" -%}
Task DSL:

{{TASK_DSL}}

Plan JSON:
{%- else -%}
Task DSL:

{{TASK_DSL}}

Return:
{"sql":"SELECT ... AS \"{{ answer.get("name", "answer") }}\" FROM ..."}
{%- endif %}
