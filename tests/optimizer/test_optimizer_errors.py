from __future__ import annotations

import unittest

from clover.optimizer import DocumentPlanParseError, OptimizerParseError, SqlParseError


class OptimizerErrorTest(unittest.TestCase):
    def test_task_specific_parse_errors_share_optimizer_base(self) -> None:
        self.assertTrue(issubclass(SqlParseError, OptimizerParseError))
        self.assertTrue(issubclass(DocumentPlanParseError, OptimizerParseError))


if __name__ == "__main__":
    unittest.main()
