from __future__ import annotations

import os
import unittest
from unittest import mock

from clover.executor import token_count


class TokenCountTest(unittest.TestCase):
    def test_rough_token_count_is_deterministic(self) -> None:
        self.assertEqual(token_count.rough_token_count(""), 0)
        self.assertEqual(token_count.rough_token_count("abcd"), 1)
        self.assertEqual(token_count.rough_token_count("abcde"), 2)

    def test_count_tokens_uses_internal_tokenizer_when_available(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                if add_special_tokens:
                    raise AssertionError("special tokens should not be counted")
                return [ord(char) for char in text]

        with mock.patch.object(token_count, "_load_tokenizer", return_value=FakeTokenizer()):
            self.assertEqual(token_count.count_tokens("abc", tokenizer_name="fake"), 3)

    def test_count_tokens_falls_back_when_tokenizer_is_unavailable(self) -> None:
        with mock.patch.object(token_count, "_load_tokenizer", return_value=None):
            self.assertEqual(token_count.count_tokens("abcde", tokenizer_name="missing"), 2)

    def test_configured_tokenizer_name_prefers_explicit_config(self) -> None:
        with mock.patch.dict(os.environ, {"CLOVER_TOKENIZER": "env-tokenizer"}):
            self.assertEqual(
                token_count.configured_tokenizer_name({"tokenizer": "config-tokenizer"}),
                "config-tokenizer",
            )

    def test_configured_tokenizer_name_reads_environment(self) -> None:
        with mock.patch.dict(os.environ, {"CLOVER_TOKENIZER": "env-tokenizer"}, clear=True):
            self.assertEqual(token_count.configured_tokenizer_name(), "env-tokenizer")


if __name__ == "__main__":
    unittest.main()
