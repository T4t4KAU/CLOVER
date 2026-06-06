Available globals; do not import or redefine:

```python
class JobManifest(BaseModel):
    chunk: str
    task: str
    advice: str

class JobOutput(BaseModel):
    explanation: str
    citation: Optional[str]
    answer: Optional[str]

class Job(BaseModel):
    manifest: JobManifest
    output: JobOutput
    sample: str
    include: Optional[bool] = None
```

Define:

```python
def prepare_jobs(
    context: List[str],
    prev_job_manifests: Optional[List[JobManifest]] = None,
    prev_job_outputs: Optional[List[JobOutput]] = None,
) -> List[JobManifest]:
    ...

def transform_outputs(jobs: List[Job]) -> str:
    ...
```

`context` is a list of document texts. You may call:

```python
def chunk_by_section(doc: str, max_chunk_size: int = 3000, overlap: int = 20) -> List[str]: ...
def chunk_by_page(doc: str, page_markers: Optional[List[str]] = None) -> List[str]: ...
```

Each `JobManifest` must be chunk-local. `transform_outputs` returns compact evidence.
