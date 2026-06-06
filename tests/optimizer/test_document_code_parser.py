from __future__ import annotations

import json
import unittest

from clover.optimizer import (
    DocumentPlanParseError,
    extract_document_python_code,
    parse_remote_document_code_to_logic_dag,
)


REMOTE_DSL = {
    "task_type": "document_reasoning",
    "question": "What was the FY2022 dividend payout ratio?",
    "sources": [
        {
            "id": "document_1",
            "type": "document",
            "source_type": "pdf",
            "format": "pdf",
            "schema": {
                "format": "pdf",
                "page_count": 12,
                "chunking": {
                    "chunk_count": 3,
                    "strategy": "sliding_window",
                    "unit": "char",
                    "size": 3000,
                    "overlap": 20,
                    "content": "resource_cache",
                },
            },
        }
    ],
    "answer": {"name": "answer", "type": "string"},
}


class DocumentCodeParserTest(unittest.TestCase):
    def test_parses_chunk_local_map_group_from_list_comprehension(self) -> None:
        code = """
```python
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    return [
        JobManifest(
            chunk=chunk,
            task="Extract dividend, net income, fiscal period, and units.",
            advice="Return answer, explanation, and citation if present."
        )
        for document in context
        for chunk in chunk_by_section(document)
    ]

def transform_outputs(jobs):
    return "\\n".join(str(job.output) for job in jobs if job.output.answer is not None)
```
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(dag["task_type"], "document_reasoning")
        self.assertEqual(dag["edges"], [])
        self.assertEqual(dag["static_collectors"][0]["kind"], "minions_transform_outputs")
        self.assertIn("def transform_outputs", dag["static_collectors"][0]["source"])
        self.assertEqual(
            dag["resource_processing"],
            [
                {
                    "id": "RP0",
                    "op": "chunk_by_section",
                    "source": "document_1",
                    "output": "CV0",
                    "params": {"max_chunk_size": 3000, "overlap": 20},
                }
            ],
        )
        self.assertEqual(
            dag["map_groups"],
            [
                {
                    "id": "G0",
                    "op": "map",
                    "inputs": {"resource_view": "CV0", "chunks": "all"},
                    "params": {
                        "local_instruction": (
                            "Extract dividend, net income, fiscal period, and units."
                        ),
                        "local_guidance": (
                            "Return answer, explanation, and citation if present."
                        ),
                        "output_contract": {
                            "format": "json",
                            "fields": ["answer", "explanation", "citation"],
                        },
                    },
                }
            ],
        )
        params = dag["map_groups"][0]["params"]
        self.assertNotIn("task", params)
        self.assertNotIn("advice", params)
        self.assertNotIn("JobManifest", json.dumps(dag))

    def test_parses_document_worker_instruction_list_cross_product(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    job_manifests = []
    instructions = [
        "Extract depreciation and amortization for FY2015.",
        "Extract total revenue for FY2015.",
    ]
    for document in context:
        chunks = chunk_by_section(document, max_chunk_size=1200, overlap=50)
        for chunk in chunks:
            for instruction in instructions:
                job_manifests.append(JobManifest(
                    chunk=chunk,
                    task=instruction,
                    advice="Return none if not present."
                ))
    return job_manifests

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            dag["resource_processing"],
            [
                {
                    "id": "RP0",
                    "op": "chunk_by_section",
                    "source": "document_1",
                    "output": "CV0",
                    "params": {"max_chunk_size": 1200, "overlap": 50},
                }
            ],
        )
        self.assertEqual(
            [group["params"]["local_instruction"] for group in dag["map_groups"]],
            [
                "Extract depreciation and amortization for FY2015.",
                "Extract total revenue for FY2015.",
            ],
        )
        self.assertEqual(
            [group["inputs"] for group in dag["map_groups"]],
            [
                {"resource_view": "CV0", "chunks": "all"},
                {"resource_view": "CV0", "chunks": "all"},
            ],
        )

    def test_parses_task_advice_pairs_and_page_chunk_alias_generator(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    job_manifests = []
    task_specs = [
        ("Extract FY2023 total revenue.", "Preserve units and citation."),
        ("Extract FY2023 depreciation and amortization.", "Return null if absent."),
    ]
    chunks = chunk_on_pages(context[0])
    job_manifests.extend(
        JobManifest(chunk=chunk, task=task, advice=advice)
        for chunk in chunks
        for task, advice in task_specs
    )
    return job_manifests

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            dag["resource_processing"],
            [
                {
                    "id": "RP0",
                    "op": "chunk_by_page",
                    "source": "document_1",
                    "output": "CV0",
                    "params": {},
                }
            ],
        )
        self.assertEqual(
            [
                (
                    group["params"]["local_instruction"],
                    group["params"]["local_guidance"],
                )
                for group in dag["map_groups"]
            ],
            [
                ("Extract FY2023 total revenue.", "Preserve units and citation."),
                (
                    "Extract FY2023 depreciation and amortization.",
                    "Return null if absent.",
                ),
            ],
        )

    def test_parses_simplified_document_worker_job_shape(self) -> None:
        code = """
def prepare_jobs(ctx, last_jobs=None):
    jobs = []
    instructions = ["Extract total revenue for FY2015."]
    chunks = chunk_by_page(ctx)
    for chunk in chunks:
        for instr in instructions:
            jobs.append(Job(instruction=instr, chunk=chunk))
    return jobs

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            dag["resource_processing"],
            [
                {
                    "id": "RP0",
                    "op": "chunk_by_page",
                    "source": "document_1",
                    "output": "CV0",
                    "params": {},
                }
            ],
        )
        self.assertEqual(
            dag["map_groups"][0]["params"]["local_instruction"],
            "Extract total revenue for FY2015.",
        )
        self.assertEqual(dag["map_groups"][0]["params"]["local_guidance"], "")

    def test_parses_append_loop_and_keeps_groups_by_instruction(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    manifests = []
    for document in context:
        chunks = chunk_by_section(document)
        for index, chunk in enumerate(chunks):
            manifests.append(JobManifest(
                chunk,
                "Extract income statement values.",
                "Use exact values and citations."
            ))
        for chunk in chunks:
            manifests.append(JobManifest(
                chunk=chunk,
                task="Extract cash flow statement values.",
                advice="Use exact values and citations."
            ))
    return manifests

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            [group["params"]["local_instruction"] for group in dag["map_groups"]],
            ["Extract income statement values.", "Extract cash flow statement values."],
        )
        self.assertEqual(
            [group["inputs"]["chunks"] for group in dag["map_groups"]],
            ["all", "all"],
        )

    def test_parses_specific_chunk_selectors_without_zero_padded_aliases(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    chunks = chunk_by_section(context[0])
    return [
        JobManifest(
            chunk=chunks[1],
            task="Inspect the relevant page window.",
            advice="Return explicit evidence only."
        ),
        JobManifest(
            chunk=chunks[2],
            task="Inspect the relevant page window.",
            advice="Return explicit evidence only.",
            chunk_id="chunk_0002"
        ),
    ]

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(dag["map_groups"][0]["inputs"]["chunks"], ["chunk_1", "chunk_2"])

    def test_preserves_replicated_job_manifest_sample_count(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    job_manifests = []
    for document in context:
        chunks = chunk_by_section(document)
        for chunk in chunks:
            job_manifest = JobManifest(
                chunk=chunk,
                task="Extract revenue evidence.",
                advice="Return explicit evidence only."
            )
            job_manifests.extend([job_manifest] * 3)
    return job_manifests

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(dag["map_groups"][0]["inputs"]["chunks"], "all")
        self.assertEqual(dag["map_groups"][0]["replicas"], 3)

    def test_parses_appended_manifest_variable(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    job_manifests = []
    for document in context:
        chunks = chunk_by_section(document)
        for chunk in chunks:
            manifest = JobManifest(
                chunk=chunk,
                task="Extract revenue evidence.",
                advice="Return explicit evidence only."
            )
            job_manifests.append(manifest)
    return job_manifests

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(dag["map_groups"][0]["inputs"]["chunks"], "all")
        self.assertEqual(
            dag["map_groups"][0]["params"]["local_instruction"],
            "Extract revenue evidence.",
        )

    def test_requires_transform_outputs(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    return [JobManifest(chunk=chunk, task="Inspect.", advice="") for chunk in context]
"""

        with self.assertRaisesRegex(DocumentPlanParseError, "transform_outputs"):
            parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

    def test_resolves_literal_worker_instruction_aliases(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    instruction = "Inspect."
    guidance = "Return explicit evidence only."
    chunks = chunk_by_section(context[0])
    return [
        JobManifest(chunk=chunk, task=instruction, advice=guidance)
        for chunk in chunks
    ]

def transform_outputs(jobs):
    return ""
"""

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            dag["map_groups"][0]["params"]["local_instruction"],
            "Inspect.",
        )
        self.assertEqual(
            dag["map_groups"][0]["params"]["local_guidance"],
            "Return explicit evidence only.",
        )

    def test_resolves_stripped_literal_worker_instruction_aliases(self) -> None:
        code = '''
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    chunks = chunk_by_section(context[0])
    for chunk in chunks:
        task = """
        Inspect explicit numerical facts in this chunk.
        """.strip()
        advice = """
        Return explicit evidence only.
        """.strip()
        return [JobManifest(chunk=chunk, task=task, advice=advice)]

def transform_outputs(jobs):
    return ""
'''

        dag = parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

        self.assertEqual(
            dag["map_groups"][0]["params"]["local_instruction"],
            "Inspect explicit numerical facts in this chunk.",
        )
        self.assertEqual(
            dag["map_groups"][0]["params"]["local_guidance"],
            "Return explicit evidence only.",
        )

    def test_rejects_dynamic_worker_instruction(self) -> None:
        code = """
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    chunks = chunk_by_section(context[0])
    return [
        JobManifest(chunk=chunk, task=build_instruction(), advice="")
        for chunk in chunks
    ]

def transform_outputs(jobs):
    return ""
"""

        with self.assertRaisesRegex(DocumentPlanParseError, "literal string"):
            parse_remote_document_code_to_logic_dag(code, REMOTE_DSL)

    def test_extracts_python_code_block(self) -> None:
        code = extract_document_python_code(
            "Here is the code:\n```python\n"
            "def prepare_jobs(context):\n    return []\n\n"
            "def transform_outputs(jobs):\n    return ''\n```"
        )

        self.assertTrue(code.startswith("def prepare_jobs"))
        self.assertIn("def transform_outputs", code)


if __name__ == "__main__":
    unittest.main()
