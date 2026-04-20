from __future__ import annotations

import unittest

from app.modules.intent_instruction.providers.base import LlmProvider
from app.modules.intent_instruction.providers.mock import MockLlmProvider


class LlmProviderProtocolTests(unittest.TestCase):
    def test_mock_llm_provider_satisfies_protocol(self):
        provider: LlmProvider = MockLlmProvider()
        self.assertTrue(hasattr(provider, "label"))
        self.assertTrue(hasattr(provider, "route"))
        self.assertTrue(hasattr(provider, "extract"))
        self.assertEqual(provider.label, "mock_llm")


if __name__ == "__main__":
    unittest.main()
