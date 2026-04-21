from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.modules.chatbox.schemas import ChatMessagePayload
from app.modules.intent_instruction.providers.mock import MockLlmProvider
from app.modules.intent_instruction.service import IntentInstructionProvider
from app.modules.intent_instruction.schemas import DeltaOperation, InstructionDelta, RouterResult
from app.shared.schemas import InstructionSnapshot


def _payload(
    message: str,
    *,
    selected=("p1", "p2"),
    unselected=("p3",),
    groups=(),
    label_context=None,
    ) -> ChatMessagePayload:
    default_label_context = {
        "analysis_context": {
            "feature_names": ["sepal_length", "petal_length"],
            "cluster_ids": ["cluster_1", "cluster_2", "cluster_3"],
            "outlier_point_ids": ["p3"],
        }
    }
    merged_label_context = dict(default_label_context)
    merged_label_context.update(label_context or {})
    return ChatMessagePayload(
        message=message,
        dataset_id="intent_service_test",
        selected_point_ids=selected,
        unselected_point_ids=unselected,
        selection_groups=groups,
        label_context=merged_label_context,
        history_window=(),
    )


@dataclass
class SelectedPointsReclassifyLlm:
    label: str = "selected-points-reclassify"

    def route(self, message, context, history, memory_context=None):
        return RouterResult(category="on_topic_actionable", confidence=0.9, reason="test")

    def extract(self, message, context, history, current_instruction, memory_context=None):
        return InstructionDelta(
            operations=(
                DeltaOperation(
                    op="add",
                    constraint_id="pending",
                    intent="reclassify_outlier",
                    payload={
                        "anchor": {"source": "selected_points", "ref": "current"},
                        "is_outlier": True,
                    },
                ),
            )
        )


@dataclass
class AmbiguousSplitLlm:
    label: str = "ambiguous-split"

    def route(self, message, context, history, memory_context=None):
        return RouterResult(
            category="on_topic_ambiguous",
            confidence=0.82,
            reason="missing target cluster",
            clarification_question="Which cluster would you like me to split?",
        )

    def extract(self, message, context, history, current_instruction, memory_context=None):
        return InstructionDelta(
            operations=(
                DeltaOperation(
                    op="add",
                    constraint_id="pending",
                    intent="split_cluster",
                    payload={"target_groups": []},
                ),
            )
        )


class IntentInstructionProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = IntentInstructionProvider(
            llm=MockLlmProvider(),
            feature_name_lookup=lambda dataset_id: ("sepal_length", "petal_length"),
        )

    def test_satisfies_intent_provider_surface(self):
        self.assertTrue(hasattr(self.provider, "respond"))
        self.assertTrue(hasattr(self.provider, "current_snapshot"))
        self.assertTrue(hasattr(self.provider, "reset"))
        self.assertIn("intent_instruction", self.provider.label)

    def test_current_snapshot_is_instruction_snapshot(self):
        snapshot = self.provider.current_snapshot("intent_service_test")
        self.assertIsInstance(snapshot, InstructionSnapshot)
        self.assertEqual(snapshot.version, 0)

    def test_off_topic_message_does_not_mutate_state(self):
        response = self.provider.respond(_payload("what's the weather?"))
        self.assertEqual(response.router_category, "off_topic")
        self.assertIsNone(response.delta)
        self.assertEqual(response.current_instruction_version, 0)

    def test_meta_query_does_not_mutate_state(self):
        response = self.provider.respond(_payload("how many clusters are there?"))
        self.assertEqual(response.router_category, "meta_query")
        self.assertIsNone(response.delta)
        self.assertEqual(response.current_instruction_version, 0)
        self.assertIn("3 clusters", response.reply)
        self.assertIn("cluster_1", response.reply)

    def test_ambiguous_message_returns_clarification(self):
        response = self.provider.respond(
            _payload("push these apart", selected=(), unselected=("p1", "p2"))
        )
        self.assertEqual(response.router_category, "on_topic_ambiguous")
        self.assertTrue(response.requires_followup)
        self.assertEqual(response.current_instruction_version, 0)

    def test_actionable_message_emits_delta_and_advances_version(self):
        response = self.provider.respond(_payload("make petal_length more important"))
        self.assertEqual(response.router_category, "on_topic_actionable")
        self.assertEqual(response.intent_type, "feature_weight")
        self.assertIsNotNone(response.delta)
        self.assertEqual(response.current_instruction_version, 1)
        snapshot = self.provider.current_snapshot("intent_service_test")
        self.assertEqual(snapshot.version, 1)
        self.assertEqual(len(snapshot.constraints), 1)

    def test_path_b_only_intent_is_compiled_without_strategy_note(self):
        response = self.provider.respond(_payload("split cluster 2 into two"))
        self.assertEqual(response.intent_type, "split_cluster")
        self.assertNotIn("deferred", response.reply.lower())

    def test_incomplete_grounding_returns_clarification_without_commit(self):
        response = self.provider.respond(
            _payload(
                "move these together",
                selected=(),
                unselected=("p1", "p2", "p3"),
            )
        )
        self.assertEqual(response.router_category, "on_topic_ambiguous")
        self.assertTrue(response.requires_followup)
        self.assertIsNone(response.delta)
        self.assertEqual(response.current_instruction_version, 0)

    def test_merge_clusters_uses_grounded_analysis_context(self):
        response = self.provider.respond(_payload("merge clusters 1 and 2"))
        self.assertEqual(response.intent_type, "merge_clusters")
        self.assertEqual(
            response.delta["operations"][0]["payload"]["target_groups"],
            [
                {"source": "cluster", "ref": "cluster_1"},
                {"source": "cluster", "ref": "cluster_2"},
            ],
        )

    def test_reclassify_selected_points_requires_single_point_grounding(self):
        provider = IntentInstructionProvider(llm=SelectedPointsReclassifyLlm())
        response = provider.respond(
            _payload(
                "the selected point should be an outlier",
                selected=("alpha_01", "outlier_north"),
                unselected=("beta_01",),
            )
        )
        self.assertEqual(response.router_category, "on_topic_ambiguous")
        self.assertIsNone(response.delta)
        self.assertIn("alpha_01", response.reply)
        self.assertIn("outlier_north", response.reply)
        self.assertEqual(response.current_instruction_version, 0)

    def test_collective_no_outlier_expands_to_per_point_ops(self):
        @dataclass
        class CollectiveReclassifyLlm:
            label: str = "collective-reclassify"

            def route(self, message, context, history, memory_context=None):
                return RouterResult(category="on_topic_actionable", confidence=0.9, reason="test")

            def extract(self, message, context, history, current_instruction, memory_context=None):
                return InstructionDelta(
                    operations=(
                        DeltaOperation(
                            op="add",
                            constraint_id="pending",
                            intent="reclassify_outlier",
                            payload={
                                "anchor": {"source": "selected_points", "ref": "current"},
                                "is_outlier": False,
                            },
                        ),
                    )
                )

        provider = IntentInstructionProvider(llm=CollectiveReclassifyLlm())
        response = provider.respond(
            _payload(
                "there should be no outlier",
                selected=("beta_01", "beta_02", "beta_03"),
                unselected=("alpha_01",),
                label_context={
                    "analysis_context": {
                        "feature_names": ["sepal_length", "petal_length"],
                        "cluster_ids": ["cluster_1", "cluster_2", "cluster_3"],
                        "outlier_point_ids": ["beta_03"],
                        "selected_point_clusters": [
                            {"point_id": "beta_01", "cluster_id": "cluster_2", "is_outlier": False},
                            {"point_id": "beta_02", "cluster_id": "cluster_2", "is_outlier": False},
                            {"point_id": "beta_03", "cluster_id": "cluster_2", "is_outlier": True},
                        ],
                    }
                },
            )
        )
        self.assertEqual(response.router_category, "on_topic_actionable")
        self.assertIsNotNone(response.delta)
        operations = response.delta["operations"]
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["intent"], "reclassify_outlier")
        self.assertEqual(operations[0]["payload"]["anchor"], {"source": "point_id", "ref": "beta_03"})
        self.assertFalse(operations[0]["payload"]["is_outlier"])

    def test_ambiguous_router_path_still_builds_draft_trace(self):
        provider = IntentInstructionProvider(llm=AmbiguousSplitLlm())
        response = provider.respond(_payload("split this cluster"))
        trace = provider.trace_for("intent_service_test")
        self.assertEqual(response.router_category, "on_topic_ambiguous")
        self.assertTrue(response.requires_followup)
        self.assertIsNone(response.delta)
        self.assertEqual(
            trace["proposed_delta"]["operations"][0]["intent"],
            "split_cluster",
        )
        self.assertEqual(trace["reply_source"], "clarification_from_draft_or_router")

    def test_reset_clears_per_dataset_state(self):
        self.provider.respond(_payload("merge clusters 1 and 2"))
        self.provider.respond(_payload("ignore cluster 3"))
        version_before = self.provider.current_snapshot("intent_service_test").version
        self.assertEqual(version_before, 2)

        self.provider.reset("intent_service_test")
        snapshot = self.provider.current_snapshot("intent_service_test")
        self.assertEqual(snapshot.version, 0)
        self.assertEqual(len(snapshot.constraints), 0)

    def test_multiple_datasets_have_independent_state(self):
        self.provider.respond(_payload("merge clusters 1 and 2"))
        other_payload = ChatMessagePayload(
            message="make petal_length more important",
            dataset_id="intent_service_other",
            selected_point_ids=(),
            unselected_point_ids=(),
            selection_groups=(),
            label_context={},
            history_window=(),
        )
        self.provider.respond(other_payload)
        self.assertEqual(self.provider.current_snapshot("intent_service_test").version, 1)
        self.assertEqual(self.provider.current_snapshot("intent_service_other").version, 1)


if __name__ == "__main__":
    unittest.main()
