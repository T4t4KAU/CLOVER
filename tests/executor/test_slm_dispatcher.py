from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from clover.executor.agents.template_tree import (
    DOCUMENT_WORKER_LEAF_KEY,
    TABLE_NUMBER_LEAF_KEY,
)
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_TPTT_LEAF_SEQUENCES_PER_TREE,
    LocalSlmSequenceDispatcher,
    LocalSlmSequenceRequest,
)
from clover.supervisor.client import RemoteLLMResult


class LocalSlmSequenceDispatcherTest(unittest.TestCase):
    def test_default_parallelism_and_tptt_epoch_cap_match_runtime_defaults(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher()
        try:
            self.assertEqual(
                dispatcher._max_parallel_sequences,  # noqa: SLF001
                DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
            )
            self.assertEqual(
                dispatcher._max_tptt_leaf_sequences_per_tree,  # noqa: SLF001
                DEFAULT_MAX_TPTT_LEAF_SEQUENCES_PER_TREE,
            )
        finally:
            dispatcher.close()

    def test_refill_sorts_queued_sequences_by_leaf_and_length(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(max_parallel_sequences=1)
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            if prompt == "hold":
                started.set()
                release.wait(timeout=2)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        dispatcher,
                        "hold",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=100,
                    ),
                ]
                self.assertTrue(started.wait(timeout=2))
                threads.extend(
                    [
                        _submit_thread(
                            dispatcher,
                            "long",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=30,
                        ),
                        _submit_thread(
                            dispatcher,
                            "short",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=5,
                        ),
                        _submit_thread(
                            dispatcher,
                            "doc",
                            leaf_key=DOCUMENT_WORKER_LEAF_KEY,
                            prompt_len=1,
                        ),
                    ]
                )
                _wait_pending(dispatcher, 3)
                release.set()
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(order, ["hold", "short", "long", "doc"])

    def test_tptt_coalescing_sorts_sequences_before_initial_dispatch(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(
            max_parallel_sequences=1,
            tptt_coalesce_ms=30.0,
        )
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        dispatcher,
                        "long",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=100,
                    ),
                    _submit_thread(
                        dispatcher,
                        "short",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=1,
                    ),
                ]
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(order, ["short", "long"])

    def test_tptt_coalesces_new_sequences_while_other_sequence_is_inflight(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(
            max_parallel_sequences=2,
            tptt_coalesce_ms=50.0,
        )
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            if prompt == "hold":
                started.set()
                release.wait(timeout=2)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                hold = _submit_thread(
                    dispatcher,
                    "hold",
                    leaf_key=DOCUMENT_WORKER_LEAF_KEY,
                    prompt_len=10,
                )
                self.assertTrue(started.wait(timeout=2))
                long = _submit_thread(
                    dispatcher,
                    "long",
                    leaf_key=TABLE_NUMBER_LEAF_KEY,
                    prompt_len=100,
                )
                time.sleep(0.01)
                short = _submit_thread(
                    dispatcher,
                    "short",
                    leaf_key=TABLE_NUMBER_LEAF_KEY,
                    prompt_len=1,
                )
                long.join(timeout=2)
                short.join(timeout=2)
                self.assertFalse(long.is_alive())
                self.assertFalse(short.is_alive())
                release.set()
                hold.join(timeout=2)
                self.assertFalse(hold.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(order, ["hold", "short", "long"])

    def test_tptt_epoch_cap_prevents_later_short_jobs_from_cutting_tree(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(
            max_parallel_sequences=1,
            max_tptt_leaf_sequences_per_tree=2,
            tptt_coalesce_ms=0.0,
        )
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            if prompt == "hold":
                started.set()
                release.wait(timeout=2)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        dispatcher,
                        "hold",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=10,
                    ),
                ]
                self.assertTrue(started.wait(timeout=2))
                threads.extend(
                    [
                        _submit_thread(
                            dispatcher,
                            "first-tree-long",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=100,
                        ),
                        _submit_thread(
                            dispatcher,
                            "second-tree-short",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=1,
                        ),
                    ]
                )
                _wait_pending(dispatcher, 2)
                release.set()
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(order, ["hold", "first-tree-long", "second-tree-short"])

    def test_same_prefix_signature_runs_before_shorter_different_prefix(self) -> None:
        tree = LocalSlmSequenceDispatcher(
            max_parallel_sequences=1,
            tptt_coalesce_ms=0.0,
        )
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            if prompt == "hold":
                started.set()
                release.wait(timeout=2)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        tree,
                        "hold",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=10,
                    ),
                ]
                self.assertTrue(started.wait(timeout=2))
                threads.extend(
                    [
                        _submit_thread(
                            tree,
                            "same-prefix-long",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=100,
                            prefix_signature="a",
                        ),
                        _submit_thread(
                            tree,
                            "same-prefix-short",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=80,
                            prefix_signature="a",
                        ),
                        _submit_thread(
                            tree,
                            "different-prefix-short",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=1,
                            prefix_signature="b",
                        ),
                    ]
                )
                _wait_pending(tree, 3)
                release.set()
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            tree.close()

        self.assertEqual(
            order,
            ["hold", "same-prefix-short", "same-prefix-long", "different-prefix-short"],
        )

    def test_fifo_slm_scheduler_preserves_fifo_order(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config={"slm_scheduler": "fifo"},
            max_parallel_sequences=1,
        )
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            del kwargs
            order.append(prompt)
            if prompt == "hold":
                started.set()
                release.wait(timeout=2)
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        dispatcher,
                        "hold",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=100,
                    ),
                ]
                self.assertTrue(started.wait(timeout=2))
                threads.extend(
                    [
                        _submit_thread(
                            dispatcher,
                            "long",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=30,
                        ),
                        _submit_thread(
                            dispatcher,
                            "short",
                            leaf_key=TABLE_NUMBER_LEAF_KEY,
                            prompt_len=5,
                        ),
                        _submit_thread(
                            dispatcher,
                            "doc",
                            leaf_key=DOCUMENT_WORKER_LEAF_KEY,
                            prompt_len=1,
                        ),
                    ]
                )
                _wait_pending(dispatcher, 3)
                release.set()
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(order, ["hold", "long", "short", "doc"])

    def test_unknown_slm_scheduler_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LocalSlmSequenceDispatcher(slm_config={"slm_scheduler": "unknown"})

    def test_unknown_leaf_is_rejected(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(max_parallel_sequences=1)
        try:
            with self.assertRaises(KeyError):
                dispatcher.generate(
                    LocalSlmSequenceRequest(
                        prompt="x",
                        leaf_key=("unknown",),
                        prompt_kind="unknown",
                    )
                )
        finally:
            dispatcher.close()

    def test_max_parallel_sequences_limits_inflight_generation(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(max_parallel_sequences=2)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_generate(prompt: str, **kwargs: object) -> RemoteLLMResult:
            nonlocal active, max_active
            del kwargs
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return _remote_result(prompt)

        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=fake_generate,
            ):
                threads = [
                    _submit_thread(
                        dispatcher,
                        f"prompt-{index}",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_len=index + 1,
                    )
                    for index in range(4)
                ]
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
        finally:
            dispatcher.close()

        self.assertEqual(max_active, 2)

    def test_result_exposes_sequence_trace_metadata(self) -> None:
        dispatcher = LocalSlmSequenceDispatcher(max_parallel_sequences=1)
        try:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                return_value=_remote_result("ok"),
            ):
                result = dispatcher.generate(
                    LocalSlmSequenceRequest(
                        prompt="prompt",
                        leaf_key=TABLE_NUMBER_LEAF_KEY,
                        prompt_kind="table_reasoning_agent_loop",
                        node_id="N0",
                        job_id="N0",
                        iteration=2,
                        prompt_len=9,
                        payload_len=9,
                    )
                )
        finally:
            dispatcher.close()

        self.assertEqual(result.text, "ok")
        trace = result.trace_metadata()
        self.assertEqual(trace["leaf_key"], list(TABLE_NUMBER_LEAF_KEY))
        self.assertEqual(trace["prompt_kind"], "table_reasoning_agent_loop")
        self.assertEqual(trace["node_id"], "N0")
        self.assertEqual(trace["iteration"], 2)
        self.assertEqual(trace["prompt_len"], 9)
        self.assertIn("prefix_signature", trace)
        self.assertIn("prefix_token_count", trace)
        self.assertEqual(trace["slm_scheduler"], "tptt")
        self.assertEqual(trace["tptt_epoch"], 1)
        self.assertIn("queue_wait_ms", trace)
        self.assertIn("inference_ms", trace)


def _submit_thread(
    dispatcher: LocalSlmSequenceDispatcher,
    prompt: str,
    *,
    leaf_key: tuple[str, ...],
    prompt_len: int,
    prefix_signature: str = "",
) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: dispatcher.generate(
            LocalSlmSequenceRequest(
                prompt=prompt,
                leaf_key=leaf_key,
                prompt_kind="table_reasoning_agent_loop",
                prompt_len=prompt_len,
                payload_len=prompt_len,
                prefix_signature=prefix_signature,
            )
        )
    )
    thread.start()
    return thread


def _wait_pending(dispatcher: LocalSlmSequenceDispatcher, expected: int) -> None:
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        with dispatcher._condition:  # noqa: SLF001 - white-box scheduler test.
            if dispatcher._pending_count >= expected:  # noqa: SLF001
                return
        time.sleep(0.01)
    raise AssertionError(f"dispatcher did not reach pending_count={expected}")


def _remote_result(text: str) -> RemoteLLMResult:
    return RemoteLLMResult(
        text=text,
        response_payload={
            "id": f"response_{text}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        response_id=f"response_{text}",
        response_status="completed",
        api_type="chat_completions",
    )


if __name__ == "__main__":
    unittest.main()
