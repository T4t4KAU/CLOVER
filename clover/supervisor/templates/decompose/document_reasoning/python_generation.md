Task DSL:

{{TASK_DSL}}

Guidance:

- Extract numbers, periods, units, line items, formulas, and citations.
- Workers do not infer across chunks; Remote synthesizes later.
- Empty chunk evidence: `answer=None`, `citation=None`.
- Use `round_state` to focus retries when present.
- No imports. Keep `task` and `advice` as literal strings or vars assigned from literal strings.

Canonical code shape:

```python
FINAL = False

def prepare_jobs(
    context: List[str],
    prev_job_manifests: Optional[List[JobManifest]] = None,
    prev_job_outputs: Optional[List[JobOutput]] = None,
) -> List[JobManifest]:
    job_manifests = []
    for document in context:
        chunks = chunk_by_section(document, max_chunk_size=3000, overlap=20)
        task = "Extract explicitly stated numerical facts relevant to the question from this chunk. Return null if absent."
        advice = "Include fiscal periods, units, line item names, and a short citation when present. Do not combine evidence across chunks."
        for chunk in chunks:
            job_manifests.append(JobManifest(chunk=chunk, task=task, advice=advice))
    return job_manifests

def transform_outputs(jobs: List[Job]) -> str:
    evidence = []
    for job in jobs:
        output = job.output
        if output.answer is not None or output.citation is not None:
            evidence.append(
                f"answer: {output.answer}\n"
                f"citation: {output.citation}\n"
                f"explanation: {output.explanation}"
            )
    return "\n\n".join(evidence)
```

Return the Python block only.
