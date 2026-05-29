from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from clover.remote_llm import create_remote_llm_session, generate_remote_text


class RemoteLLMClientTest(unittest.TestCase):
    def test_generates_text_with_responses_api(self) -> None:
        result = generate_remote_text(
            prompt="hello",
            remote_config={"api_type": "responses", "model": "fake-model"},
            client=_FakeResponsesClient("SELECT 1"),
        )

        self.assertEqual(result.text, "SELECT 1")
        self.assertEqual(result.response_id, "resp_fake")
        self.assertEqual(result.response_status, "completed")
        self.assertEqual(result.api_type, "responses")

    def test_resolves_remote_model_from_environment_reference(self) -> None:
        client = _FakeResponsesClient("SELECT 1")
        with patch.dict(
            "os.environ",
            {"TEST_CLOVER_REMOTE_MODEL": "env-remote-model"},
            clear=False,
        ):
            generate_remote_text(
                prompt="hello",
                remote_config={
                    "api_type": "responses",
                    "model_env": "TEST_CLOVER_REMOTE_MODEL",
                },
                client=client,
            )

        self.assertEqual(client.responses.last_request["model"], "env-remote-model")

    def test_generates_text_with_chat_completions_api(self) -> None:
        client = _FakeChatClient("SELECT 2")
        result = generate_remote_text(
            prompt="hello",
            remote_config={
                "api_type": "chat_completions",
                "model": "fake-model",
                "max_tokens": 1024,
                "temperature": 0,
                "reasoning_effort": "high",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
            client=client,
        )

        self.assertEqual(result.text, "SELECT 2")
        self.assertEqual(result.response_id, "chat_fake")
        self.assertEqual(result.response_status, "completed")
        self.assertEqual(result.api_type, "chat_completions")
        self.assertEqual(
            client.chat.completions.last_request["extra_body"],
            {"thinking": {"type": "enabled"}},
        )
        self.assertEqual(client.chat.completions.last_request["reasoning_effort"], "high")

    def test_responses_session_reuses_previous_response_id(self) -> None:
        client = _StatefulResponsesClient(["first", "second"])
        session = create_remote_llm_session(
            {"api_type": "responses", "model": "fake-model"},
            client=client,
        )

        first = session.generate("task dsl")
        second = session.generate("report")

        self.assertEqual(first.response_id, "resp_0")
        self.assertEqual(second.response_id, "resp_1")
        self.assertIsNone(client.responses.requests[0].get("previous_response_id"))
        self.assertEqual(client.responses.requests[1]["previous_response_id"], "resp_0")

    def test_chat_session_reuses_message_history(self) -> None:
        client = _StatefulChatClient(["sql", '{"answer": 2, "retry": false, "new_sql": null}'])
        session = create_remote_llm_session(
            {
                "api_type": "chat_completions",
                "model": "fake-model",
                "system_message": "shared session",
            },
            client=client,
        )

        session.generate("task dsl")
        session.generate("report")

        second_messages = client.chat.completions.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(second_messages[1]["content"], "task dsl")
        self.assertEqual(second_messages[2]["content"], "sql")
        self.assertEqual(second_messages[3]["content"], "report")


class _FakeResponsesClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.last_request: dict[str, object] = {}

    def create(self, **kwargs: object) -> "_FakeResponsesResponse":
        self.last_request = kwargs
        return _FakeResponsesResponse(self._output_text)


class _StatefulResponsesClient:
    def __init__(self, output_texts: list[str]) -> None:
        self.responses = _StatefulResponses(output_texts)


class _StatefulResponses:
    def __init__(self, output_texts: list[str]) -> None:
        self._output_texts = output_texts
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeResponsesResponse":
        self.requests.append(kwargs)
        index = len(self.requests) - 1
        return _FakeResponsesResponse(self._output_texts[index], response_id=f"resp_{index}")


class _FakeResponsesResponse:
    status = "completed"

    def __init__(self, output_text: str, response_id: str = "resp_fake") -> None:
        self.id = response_id
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=output_text)],
            )
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "status": self.status, "mode": mode}


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


class _StatefulChatClient:
    def __init__(self, output_texts: list[str]) -> None:
        self.chat = SimpleNamespace(
            completions=_StatefulChatCompletions(output_texts),
        )


class _StatefulChatCompletions:
    def __init__(self, output_texts: list[str]) -> None:
        self._output_texts = output_texts
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        return _FakeChatResponse(self._output_texts[len(self.requests) - 1])


class _FakeChatResponse:
    id = "chat_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
