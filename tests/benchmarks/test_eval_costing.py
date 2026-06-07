from __future__ import annotations

import unittest

from benchmarks.costing import (
    DEFAULT_OPENAI_REFERENCE_MODEL,
    estimate_openai_text_cost,
    normalize_remote_token_usage,
)


class EvalCostingTest(unittest.TestCase):
    def test_estimates_openai_text_cost_ignores_cached_input_discount(self) -> None:
        estimate = estimate_openai_text_cost(
            {
                "input_tokens": 1_000_000,
                "cached_input_tokens": 100_000,
                "output_tokens": 500_000,
                "reasoning_tokens": 25_000,
                "total_tokens": 1_500_000,
            },
            pricing_model="gpt-5.2",
        )

        self.assertEqual(estimate["pricing_model"], "gpt-5.2")
        self.assertEqual(estimate["usage"]["cached_input_tokens"], 100_000)
        self.assertEqual(estimate["usage"]["billable_input_tokens"], 1_000_000)
        self.assertEqual(
            estimate["usage"]["cache_discount_ignored_input_tokens"],
            100_000,
        )
        self.assertAlmostEqual(estimate["cost_usd"]["input"], 1.75)
        self.assertAlmostEqual(estimate["cost_usd"]["cached_input"], 0.0)
        self.assertAlmostEqual(estimate["cost_usd"]["output"], 7.0)
        self.assertAlmostEqual(estimate["cost_usd"]["total"], 8.75)

    def test_uses_openai_remote_model_when_known(self) -> None:
        estimate = estimate_openai_text_cost(
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            remote_config={"provider": "openai", "model": "gpt-4o-mini"},
        )

        self.assertEqual(estimate["pricing_model"], "gpt-4o-mini")
        self.assertEqual(estimate["pricing_model_source"], "remote_config")
        self.assertAlmostEqual(estimate["cost_usd"]["total"], 0.75)

    def test_non_openai_remote_model_falls_back_to_reference_model(self) -> None:
        estimate = estimate_openai_text_cost(
            {"input_tokens": 1, "output_tokens": 1},
            remote_config={"provider": "doubao", "model": "doubao-seed"},
        )

        self.assertEqual(estimate["pricing_model"], DEFAULT_OPENAI_REFERENCE_MODEL)
        self.assertEqual(estimate["pricing_model_source"], "default_reference")
        self.assertEqual(estimate["remote_provider"], "doubao")

    def test_uses_deepseek_remote_pricing_when_known(self) -> None:
        estimate = estimate_openai_text_cost(
            {
                "input_tokens": 1_000_000,
                "cached_input_tokens": 100_000,
                "output_tokens": 500_000,
            },
            remote_config={"provider": "deepseek", "model": "deepseek-v4-pro"},
        )

        self.assertEqual(estimate["provider"], "deepseek")
        self.assertEqual(estimate["pricing_model"], "deepseek-v4-pro")
        self.assertEqual(estimate["usage"]["billable_input_tokens"], 1_000_000)
        self.assertEqual(
            estimate["usage"]["cache_discount_ignored_input_tokens"],
            100_000,
        )
        self.assertAlmostEqual(estimate["cost_usd"]["input"], 0.435)
        self.assertAlmostEqual(estimate["cost_usd"]["cached_input"], 0.0)
        self.assertAlmostEqual(estimate["cost_usd"]["output"], 0.435)
        self.assertAlmostEqual(estimate["cost_usd"]["total"], 0.87)

    def test_normalizes_missing_usage_fields(self) -> None:
        self.assertEqual(
            normalize_remote_token_usage({"input_tokens": 5}),
            {
                "input_tokens": 5,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
