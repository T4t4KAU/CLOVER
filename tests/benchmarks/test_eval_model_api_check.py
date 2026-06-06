from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import eval as eval_cli


class EvalModelApiCheckTest(unittest.TestCase):
    def test_preflight_checks_remote_and_local_with_small_requests(self) -> None:
        remote_config = {
            "provider": "doubao",
            "api_type": "responses",
            "base_url": "https://example.test/v1",
            "model": "remote-model",
            "timeout": 180,
            "max_retries": 2,
            "max_output_tokens": 12000,
        }
        local_config = {
            "provider": "local",
            "api_type": "chat_completions",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "timeout": 1800,
            "max_retries": 2,
            "max_tokens": 4096,
        }
        args = SimpleNamespace(
            skip_model_api_check=False,
            model_api_check_timeout=5,
        )

        with patch("benchmarks.eval.generate_remote_text") as generate:
            generate.return_value = SimpleNamespace(text="OK")
            eval_cli.preflight_model_api_checks(
                args=args,
                remote_config=remote_config,
                local_slm_config=local_config,
            )

        self.assertEqual(generate.call_count, 2)
        remote_call = generate.call_args_list[0].kwargs["remote_config"]
        local_call = generate.call_args_list[1].kwargs["remote_config"]
        self.assertEqual(remote_call["timeout"], 5)
        self.assertEqual(remote_call["max_retries"], 0)
        self.assertEqual(remote_call["max_output_tokens"], 8)
        self.assertEqual(local_call["timeout"], 5)
        self.assertEqual(local_call["max_retries"], 0)
        self.assertEqual(local_call["max_tokens"], 8)
        self.assertEqual(remote_config["timeout"], 180)
        self.assertEqual(local_config["timeout"], 1800)

    def test_preflight_can_be_skipped(self) -> None:
        args = SimpleNamespace(
            skip_model_api_check=True,
            model_api_check_timeout=5,
        )

        with patch("benchmarks.eval.generate_remote_text") as generate:
            eval_cli.preflight_model_api_checks(
                args=args,
                remote_config={"model": "remote"},
                local_slm_config={"model": "local"},
            )

        generate.assert_not_called()

    def test_preflight_failure_exits_before_eval(self) -> None:
        args = SimpleNamespace(
            skip_model_api_check=False,
            model_api_check_timeout=5,
        )

        with patch("benchmarks.eval.generate_remote_text", side_effect=RuntimeError("boom")):
            with self.assertRaises(SystemExit) as raised:
                eval_cli.preflight_model_api_checks(
                    args=args,
                    remote_config={"provider": "remote", "model": "m"},
                    local_slm_config=None,
                )

        self.assertIn("Model API connectivity check failed", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
