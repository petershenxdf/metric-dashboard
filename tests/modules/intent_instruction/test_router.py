from __future__ import annotations

import unittest

from app.modules.intent_instruction.providers.mock import MockLlmProvider
from app.modules.intent_instruction.router import classify
from app.modules.intent_instruction.schemas import DatasetContext


class MockLlmRouterTests(unittest.TestCase):
    def setUp(self):
        self.llm = MockLlmProvider()
        self.context_with_selection = DatasetContext(
            dataset_id="intent_router_test",
            feature_names=("sepal_length", "petal_length"),
            cluster_ids=("cluster_1", "cluster_2", "cluster_3"),
            selected_point_ids=("p1", "p2"),
            unselected_point_ids=("p3",),
        )
        self.context_empty_selection = DatasetContext(
            dataset_id="intent_router_test",
            feature_names=("sepal_length", "petal_length"),
            cluster_ids=("cluster_1", "cluster_2", "cluster_3"),
        )

    def test_off_topic_message(self):
        result = classify(self.llm, "today's weather?", self.context_with_selection)
        self.assertEqual(result.category, "off_topic")

    def test_meta_query(self):
        result = classify(self.llm, "how many clusters are there?", self.context_with_selection)
        self.assertEqual(result.category, "meta_query")

    def test_selection_reference_without_selection_is_ambiguous(self):
        result = classify(self.llm, "move these together", self.context_empty_selection)
        self.assertEqual(result.category, "on_topic_ambiguous")
        self.assertIsNotNone(result.clarification_question)

    def test_feature_weight_message_is_actionable(self):
        result = classify(
            self.llm,
            "make petal_length more important",
            self.context_empty_selection,
        )
        self.assertEqual(result.category, "on_topic_actionable")

    def test_merge_clusters_message_is_actionable(self):
        result = classify(self.llm, "merge clusters 1 and 2", self.context_empty_selection)
        self.assertEqual(result.category, "on_topic_actionable")

    def test_empty_message_is_ambiguous(self):
        result = classify(self.llm, "   ", self.context_empty_selection)
        self.assertEqual(result.category, "on_topic_ambiguous")

    def test_unknown_phrase_is_ambiguous(self):
        result = classify(self.llm, "zorp the flarn please", self.context_with_selection)
        self.assertEqual(result.category, "on_topic_ambiguous")


if __name__ == "__main__":
    unittest.main()
