from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.workflows.intent_runtime_support import IntentRuntimeConfig


class IntentRuntimeSupportTests(unittest.TestCase):
    def test_runtime_config_reads_defaults_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "METRIC_DASHBOARD_LLM_PROVIDER": "ollama",
                "METRIC_DASHBOARD_LLM_MODEL": "phi4-mini",
                "METRIC_DASHBOARD_OLLAMA_BASE_URL": "http://127.0.0.1:22434",
                "METRIC_DASHBOARD_OLLAMA_KEEP_ALIVE": "20m",
                "METRIC_DASHBOARD_LLM_TEMPERATURE": "0.3",
                "METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS": "60",
                "METRIC_DASHBOARD_LLM_MAX_OUTPUT_TOKENS": "1200",
                "METRIC_DASHBOARD_LLM_ALLOW_MOCK_FALLBACK": "false",
            },
            clear=False,
        ):
            config = IntentRuntimeConfig()

        self.assertEqual(config.provider_kind, "ollama")
        self.assertEqual(config.model_name, "phi4-mini")
        self.assertEqual(config.base_url, "http://127.0.0.1:22434")
        self.assertEqual(config.keep_alive, "20m")
        self.assertAlmostEqual(config.temperature, 0.3, places=2)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.max_output_tokens, 1200)
        self.assertFalse(config.allow_mock_fallback)

    def test_runtime_config_accepts_response_mode_without_provider_rebuild_fields(self):
        config = IntentRuntimeConfig()

        updated = config.merged({"response_mode": "raw"})

        self.assertEqual(updated.response_mode, "raw")
        self.assertEqual(updated.model_name, config.model_name)


if __name__ == "__main__":
    unittest.main()
