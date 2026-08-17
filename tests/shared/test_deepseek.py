import unittest
from unittest.mock import patch

from app.shared.deepseek import (
    DEEPSEEK_PRO_MODEL,
    DeepSeekClient,
    DeepSeekResponseContentError,
)


class DeepSeekClientTests(unittest.TestCase):
    def test_health_explains_when_api_key_is_missing(self):
        client = DeepSeekClient(api_key="")

        health = client.health()

        self.assertFalse(health["available"])
        self.assertEqual(health["model_name"], DEEPSEEK_PRO_MODEL)
        self.assertIn("API_KEY", health["error"])

    def test_json_generation_uses_the_constrained_v4_pro_profile(self):
        client = DeepSeekClient(api_key="test-key")
        provider_response = {
            "id": "response-1",
            "model": DEEPSEEK_PRO_MODEL,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"summary": "plain explanation"}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(client, "_post_json", return_value=provider_response) as post:
            parsed, raw, metadata = client.generate_json_with_metadata("Explain this plan")

        self.assertEqual(parsed, {"summary": "plain explanation"})
        self.assertEqual(raw, '{"summary": "plain explanation"}')
        self.assertEqual(metadata["model"], DEEPSEEK_PRO_MODEL)
        request_body = post.call_args.args[1]
        self.assertEqual(request_body["model"], DEEPSEEK_PRO_MODEL)
        self.assertEqual(request_body["temperature"], 0.0)
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["response_format"], {"type": "json_object"})

    def test_missing_message_content_preserves_provider_diagnostics(self):
        client = DeepSeekClient(api_key="test-key")
        provider_response = {
            "model": DEEPSEEK_PRO_MODEL,
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "reasoning_content": "hidden"},
                }
            ],
            "usage": {"total_tokens": 2200},
        }

        with patch.object(client, "_post_json", return_value=provider_response):
            with self.assertRaises(DeepSeekResponseContentError) as context:
                client.generate_json_with_metadata("Explain this plan")

        self.assertEqual(context.exception.response_metadata["finish_reason"], "length")
        self.assertFalse(context.exception.response_metadata["has_content"])


if __name__ == "__main__":
    unittest.main()
