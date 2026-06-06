Answer the question using only the worker evidence below.

Question:
{{ task.question }}

Worker summary:
- ok: {{ observations.ok }}
- workers: {{ observations.worker_count }}
- included: {{ observations.included_count }}
- failed: {{ observations.failed_count }}
{% if observations.error %}- error: {{ observations.error.type }}: {{ observations.error.message }}{% endif %}
{% if observations.fallback_used %}- transform_outputs fallback: true{% endif %}
{% if observations.evidence_truncated %}- evidence truncated: true{% endif %}
{% if observations.transform_error %}- transform_outputs error: {{ observations.transform_error }}{% endif %}
{% if observations.feedback %}- previous feedback: {{ observations.feedback }}{% endif %}
{% if observations.scratchpad %}- supervisor scratchpad: {{ observations.scratchpad }}{% endif %}

{% if observations.prior_evidence_summary %}
Prior worker evidence from previous rounds:
- rounds: {{ observations.prior_evidence_round_count }}
{% if observations.prior_evidence_truncated %}- prior evidence truncated: true{% endif %}

{{ observations.prior_evidence_summary }}

{% endif %}
Worker evidence:
{{ observations.evidence_summary }}

Return one JSON object only:
{% if force_final_answer %}
{"answer": "", "sufficient": true, "explanation": "", "feedback": null, "scratchpad": ""}
{% else %}
{"answer": null, "sufficient": false, "explanation": "", "feedback": "", "scratchpad": "", "next_python_code": null}
{% endif %}

Rules:
{% if force_final_answer %}
- This is the last supervisor pass. Do not request another worker round.
- Set "sufficient" to true and put the best supported answer in "answer".
- If evidence is incomplete, give the most conservative answer supported by the evidence and explain the limitation.
- Set "feedback" to null.
{% else %}
- If the evidence is sufficient, set "sufficient" to true and put the answer in "answer".
- If the evidence is insufficient or contradictory, set "sufficient" to false, set "answer" to null, write concise "feedback", and put the next worker Python code in "next_python_code".
- "next_python_code" must be a JSON string defining exactly `prepare_jobs(...)` and `transform_outputs(jobs)`. Do not put code fences inside the JSON string.
- The code may use `JobManifest`, `Job`, `JobOutput`, `chunk_by_section`, and `chunk_by_page` from global scope. Do not import or redefine them.
- `prepare_jobs` must create chunk-local atomic worker tasks. `transform_outputs` must filter and aggregate worker outputs into a compact evidence string.
{% endif %}
- Use "scratchpad" to preserve only compact progress needed by the next supervisor round.
- Use prior worker evidence and current worker evidence together; do not discard values found in previous rounds.
- Use only the evidence above.
- The "answer" field itself must be self-contained; do not put critical numbers only in "explanation".
- For numerical questions, perform the required calculation and round exactly as requested.
- If the answer is yes/no, include the decisive numerical value, change, or period when the evidence supports it.
- If the answer asks for a category or comparison, include the selected category and its decisive value when the evidence supports it.
- In "explanation", include the source values needed to verify a numerical calculation.
- Keep "explanation" concise.
