import unittest

from app.modules.algorithm_adapters.service import run_default_analysis
from app.modules.rule_panel.schemas import TreeConfig
from app.modules.rule_panel.service import generate_rule_set, interpret_rule_set_preview
from app.shared.wine_dataset import (
    WINE_DATASET_ID,
    WINE_FEATURE_NAMES,
    load_wine_dataset,
    load_wine_feature_matrix,
)


class RulePanelServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_wine_dataset()
        cls.matrix = load_wine_feature_matrix()
        cls.analysis = run_default_analysis(cls.matrix, n_clusters=3)

    def test_generates_cluster_and_anomaly_rules_from_ssdbcodi_output(self):
        rule_set = generate_rule_set(
            self.matrix,
            self.analysis,
            dataset_id=self.dataset.dataset_id,
            config=TreeConfig(max_depth=3, min_samples_leaf=1),
        )

        self.assertGreater(len(rule_set.rules), 0)
        self.assertEqual(rule_set.dataset_id, WINE_DATASET_ID)
        self.assertTrue(any(rule.target_kind == "cluster" for rule in rule_set.rules))
        self.assertTrue(any(rule.target_kind == "anomaly" for rule in rule_set.rules))
        self.assertEqual(rule_set.model["source_of_truth"], "ssdbcodi")
        self.assertEqual(rule_set.model["role"], "explanation_only")
        self.assertEqual(rule_set.diagnostics["source_point_count"], 129)
        self.assertEqual(rule_set.diagnostics["source_feature_count"], len(WINE_FEATURE_NAMES))
        self.assertIn("alcohol", rule_set.diagnostics["raw_feature_names"])
        self.assertIn("feature_usage", rule_set.diagnostics)

    def test_rule_conditions_reference_known_features_and_match_points(self):
        rule_set = generate_rule_set(
            self.matrix,
            self.analysis,
            dataset_id=self.dataset.dataset_id,
        )
        feature_names = set(self.matrix.feature_names)
        condition_features = {
            condition.feature
            for rule in rule_set.rules
            for condition in rule.conditions
        }
        rows_by_point = dict(zip(self.matrix.point_ids, self.matrix.values))
        feature_index = {name: index for index, name in enumerate(self.matrix.feature_names)}

        self.assertIn("alcohol", feature_names)
        self.assertIn("proline", feature_names)
        self.assertGreater(len(condition_features), 0)
        self.assertTrue(condition_features.issubset(set(WINE_FEATURE_NAMES)))
        self.assertFalse(condition_features.intersection({"x", "y"}))

        for rule in rule_set.rules:
            for condition in rule.conditions:
                self.assertIn(condition.feature, feature_names)
            for point_id in rule.matched_point_ids:
                row = rows_by_point[point_id]
                for condition in rule.conditions:
                    value = row[feature_index[condition.feature]]
                    if condition.operator == "<=":
                        self.assertLessEqual(value, condition.threshold)
                    else:
                        self.assertGreater(value, condition.threshold)

    def test_generation_is_deterministic_for_fixed_input(self):
        first = generate_rule_set(self.matrix, self.analysis, dataset_id=self.dataset.dataset_id)
        second = generate_rule_set(self.matrix, self.analysis, dataset_id=self.dataset.dataset_id)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_interpretation_preview_uses_allowed_categories(self):
        rule_set = generate_rule_set(self.matrix, self.analysis, dataset_id=self.dataset.dataset_id)
        interpretation = interpret_rule_set_preview(rule_set)

        self.assertIn("label_priority", interpretation.categories)
        self.assertIn("feature_label_strategy", interpretation.categories)
        self.assertIn("recommendation", interpretation.to_dict())
        self.assertGreater(len(interpretation.label_targets), 0)
        self.assertGreater(len(interpretation.suspicion_reasons), 0)
        self.assertGreater(len(interpretation.point_label_guidance), 0)
        self.assertGreater(len(interpretation.quantitative_findings), 0)
        self.assertGreater(len(interpretation.suggested_label_actions), 0)
        self.assertGreater(len(interpretation.target_rule_ids), 0)


if __name__ == "__main__":
    unittest.main()
