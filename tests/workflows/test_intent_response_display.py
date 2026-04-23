from __future__ import annotations

import unittest

from app.modules.chatbox.schemas import ChatResponse, ChatTurn, ChatboxState
from app.shared.schemas import InstructionSnapshot
from app.workflows.intent_response_display import (
    build_display_chat_state,
    build_display_response,
)


class IntentResponseDisplayTests(unittest.TestCase):
    def test_processed_mode_keeps_processed_reply(self):
        response = ChatResponse(
            reply="Processed answer",
            router_category="meta_query",
            delta=None,
            current_instruction_version=0,
            provider_label="intent_instruction(mock)",
        ).to_dict()

        display_response = build_display_response(response, {}, "processed")

        self.assertEqual(display_response["reply"], "Processed answer")
        self.assertEqual(display_response["processed_reply"], "Processed answer")
        self.assertIsNone(display_response["raw_reply"])

    def test_raw_mode_prefers_provider_trace_output(self):
        response = ChatResponse(
            reply="Processed answer",
            router_category="on_topic_actionable",
            delta={"operations": [{"op": "add", "constraint_id": "c1"}]},
            current_instruction_version=1,
            provider_label="intent_instruction(ollama:qwen2.5:14b)",
        ).to_dict()
        provider_trace = {
            "llm_trace": {
                "reply": {"raw_response": "I understand this as a request to merge cluster_1 and cluster_2."},
            }
        }

        display_response = build_display_response(response, provider_trace, "raw")

        self.assertEqual(
            display_response["reply"],
            "I understand this as a request to merge cluster_1 and cluster_2.",
        )
        self.assertEqual(display_response["processed_reply"], "Processed answer")
        self.assertTrue(display_response["raw_reply_available"])

    def test_raw_mode_builds_chat_state_from_interactions(self):
        base_state = ChatboxState(
            dataset_id="demo",
            turns=(
                ChatTurn(turn_id="turn_001", role="user", text="merge clusters 1 and 2"),
                ChatTurn(turn_id="turn_002", role="assistant", text="Okay, I recorded a merge."),
            ),
            instruction_snapshot=InstructionSnapshot(version=1, constraints=()),
        )
        interactions = [
            {
                "response": {
                    "reply": "Okay, I recorded a merge.",
                    "router_category": "on_topic_actionable",
                    "current_instruction_version": 1,
                    "delta": {"operations": [{"op": "add", "constraint_id": "c1"}]},
                    "provider_label": "intent_instruction(mock)",
                },
                "provider_trace": {
                    "llm_trace": {
                        "reply": {
                            "raw_response": "I understand this as a request to merge cluster_1 and cluster_2."
                        }
                    },
                },
            }
        ]

        display_state = build_display_chat_state(base_state, interactions, "raw")

        self.assertEqual(display_state.turns[0].text, "merge clusters 1 and 2")
        self.assertEqual(
            display_state.turns[1].text,
            "I understand this as a request to merge cluster_1 and cluster_2.",
        )


if __name__ == "__main__":
    unittest.main()
