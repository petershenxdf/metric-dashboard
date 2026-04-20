from __future__ import annotations

import unittest

from app.modules.intent_instruction.extractor import apply_delta, propose_delta
from app.modules.intent_instruction.providers.mock import MockLlmProvider
from app.modules.intent_instruction.schemas import (
    DatasetContext,
    InstructionDelta,
    StructuredInstruction,
)
from app.modules.intent_instruction.store import IntentInstructionStore


class MockLlmExtractorTests(unittest.TestCase):
    def setUp(self):
        self.llm = MockLlmProvider()
        self.context = DatasetContext(
            dataset_id="intent_ext",
            feature_names=("sepal_length", "petal_length"),
            cluster_ids=("cluster_1", "cluster_2", "cluster_3"),
            selected_point_ids=("p1", "p2"),
            unselected_point_ids=("p3",),
        )
        self.empty_state = StructuredInstruction(version=0, constraints=())

    def _propose(self, message: str) -> InstructionDelta:
        return propose_delta(self.llm, message, self.context, (), self.empty_state)

    def test_group_similar_delta(self):
        delta = self._propose("these points should be together")
        ops = delta.operations
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].intent, "group_similar")
        self.assertEqual(ops[0].op, "add")

    def test_group_dissimilar_delta(self):
        delta = self._propose("push cluster 1 away from cluster 3")
        self.assertEqual(delta.operations[0].intent, "group_dissimilar")

    def test_merge_clusters_delta(self):
        delta = self._propose("merge clusters 1 and 2")
        op = delta.operations[0]
        self.assertEqual(op.intent, "merge_clusters")
        target_groups = op.payload.get("target_groups")
        self.assertTrue(any(g["ref"] == "cluster_1" for g in target_groups))
        self.assertTrue(any(g["ref"] == "cluster_2" for g in target_groups))

    def test_feature_weight_decrease_delta(self):
        delta = self._propose("make feature sepal_length less important")
        op = delta.operations[0]
        self.assertEqual(op.intent, "feature_weight")
        self.assertEqual(op.payload["direction"], "decrease")
        self.assertEqual(op.payload["feature"], "sepal_length")

    def test_feature_weight_increase_delta(self):
        delta = self._propose("make petal_length more important")
        op = delta.operations[0]
        self.assertEqual(op.intent, "feature_weight")
        self.assertEqual(op.payload["direction"], "increase")

    def test_anchor_point_delta(self):
        delta = self._propose("treat p1 as a typical example for cluster 2")
        op = delta.operations[0]
        self.assertEqual(op.intent, "anchor_point")
        self.assertEqual(op.payload["anchor"]["ref"], "p1")

    def test_ignore_cluster_delta(self):
        delta = self._propose("ignore cluster 3")
        op = delta.operations[0]
        self.assertEqual(op.intent, "ignore_cluster")
        target_groups = op.payload["target_groups"]
        self.assertTrue(any(g["ref"] == "cluster_3" for g in target_groups))

    def test_split_cluster_delta(self):
        delta = self._propose("split cluster 2 into two")
        self.assertEqual(delta.operations[0].intent, "split_cluster")

    def test_reclassify_outlier_delta(self):
        delta = self._propose("p3 is not an outlier")
        op = delta.operations[0]
        self.assertEqual(op.intent, "reclassify_outlier")
        self.assertFalse(op.payload["is_outlier"])

    def test_applying_delta_advances_version_and_assigns_id(self):
        store = IntentInstructionStore()
        dataset_id = self.context.dataset_id
        proposed = self._propose("merge clusters 1 and 2")
        next_state, final_delta = apply_delta(
            store,
            dataset_id,
            proposed,
            raw_message="Merge clusters 1 and 2",
            router_category="on_topic_actionable",
            confidence=0.9,
        )
        self.assertEqual(next_state.version, 1)
        self.assertEqual(len(next_state.constraints), 1)
        self.assertEqual(next_state.constraints[0].intent, "merge_clusters")
        self.assertEqual(final_delta.operations[0].constraint_id, "c1")
        self.assertNotEqual(final_delta.operations[0].constraint_id, "pending")

    def test_applying_two_deltas_increments_ids_and_version(self):
        store = IntentInstructionStore()
        dataset_id = self.context.dataset_id
        first = self._propose("merge clusters 1 and 2")
        apply_delta(store, dataset_id, first, raw_message="msg", router_category="on_topic_actionable", confidence=0.9)
        second = self._propose("make petal_length more important")
        state_after, delta_after = apply_delta(
            store, dataset_id, second, raw_message="msg2", router_category="on_topic_actionable", confidence=0.9
        )
        self.assertEqual(state_after.version, 2)
        self.assertEqual(len(state_after.constraints), 2)
        self.assertEqual(delta_after.operations[0].constraint_id, "c2")

    def test_remove_operation_drops_matching_constraint(self):
        from app.modules.intent_instruction.schemas import DeltaOperation

        store = IntentInstructionStore()
        dataset_id = self.context.dataset_id
        proposed = self._propose("merge clusters 1 and 2")
        apply_delta(store, dataset_id, proposed, raw_message="m", router_category="on_topic_actionable", confidence=0.9)
        remove_delta = InstructionDelta(
            operations=(DeltaOperation(op="remove", constraint_id="c1"),)
        )
        state_after, _ = apply_delta(
            store, dataset_id, remove_delta, raw_message="remove", router_category="on_topic_actionable", confidence=0.9
        )
        self.assertEqual(len(state_after.constraints), 0)


if __name__ == "__main__":
    unittest.main()
