from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor.local_slm import generate_slm_text, load_slm_config


class SlmClientTest(unittest.TestCase):
    def test_loads_slm_config_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "slm.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "dashscope",
                        "api_type": "chat_completions",
                        "model": "qwen3.6-27b",
                    }
                ),
                encoding="utf-8",
            )

            config = load_slm_config(config_path)

        self.assertEqual(config["provider"], "dashscope")
        self.assertEqual(config["model"], "qwen3.6-27b")

    def test_loads_slm_config_fields_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "slm.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "dashscope",
                        "api_type": "chat_completions",
                        "api_key_env": "TEST_CLOVER_SLM_KEY",
                        "model": "file-model",
                        "model_env": "TEST_CLOVER_SLM_MODEL_REF",
                        "max_tokens": 1024,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "TEST_CLOVER_SLM_KEY": "env-secret",
                    "TEST_CLOVER_SLM_MODEL_REF": "env-ref-model",
                    "CLOVER_LOCAL_SLM_MODEL": "env-prefix-model",
                    "CLOVER_LOCAL_SLM_MAX_TOKENS": "2048",
                },
                clear=False,
            ):
                config = load_slm_config(config_path)

        self.assertEqual(config["api_key"], "env-secret")
        self.assertEqual(config["model"], "env-prefix-model")
        self.assertEqual(config["max_tokens"], 2048)

    def test_generates_text_with_configured_chat_completion_options(self) -> None:
        client = _FakeChatClient("我是 CLOVER 的本地节点模型。")

        result = generate_slm_text(
            "你是谁",
            slm_config={
                "api_type": "chat_completions",
                "model": "qwen3.6-27b",
                "temperature": 0,
                "max_tokens": 4096,
                "extra_body": {"enable_thinking": False},
            },
            client=client,
        )

        self.assertEqual(result.text, "我是 CLOVER 的本地节点模型。")
        self.assertEqual(
            client.chat.completions.last_request["extra_body"],
            {"enable_thinking": False},
        )
        self.assertEqual(client.chat.completions.last_request["model"], "qwen3.6-27b")


class _FakeChatClient:
    def __init__(self, output_text: str) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(output_text),
        )


class _FakeChatCompletions:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.last_request: dict[str, object] = {}

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.last_request = kwargs
        return _FakeChatResponse(self._output_text)


class _FakeChatResponse:
    id = "slm_chat_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
