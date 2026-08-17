import copy
import unittest

from app.modules.algorithm_adapters.service import run_default_analysis
from app.modules.rule_panel.recommendation import (
    build_recommendation_plan,
    build_rule_guidance_metrics,
)
from app.modules.rule_panel.schemas import (
    RECOMMENDATION_CATEGORIES,
    RecommendationPlan,
)
from app.modules.rule_panel.service import generate_rule_set
from app.shared.wine_dataset import WINE_DATASET_ID, load_wine_feature_matrix


class DeterministicRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_wine_feature_matrix()
        cls.analysis = run_default_analysis(cls.matrix, n_clusters=3)
        cls.rule_set = generate_rule_set(
            cls.matrix,
            cls.analysis,
            dataset_id=WINE_DATASET_ID,
        )
        cls.metrics = build_rule_guidance_metrics(
            cls.rule_set,
            analysis_result=cls.analysis,
            feature_matrix=cls.matrix,
        )

    def test_every_category_is_stable_for_the_same_analysis_state(self):
        for category in RECOMMENDATION_CATEGORIES:
            with self.subTest(category=category):
                first = self._build(category)
                second = self._build(category)

                self.assertEqual(first, second)
                validated = RecommendationPlan.from_dict(first)
                self.assertEqual(
                    validated.highlighted_point_ids,
                    validated.recommended_point_ids,
                )
                self.assertTrue(
                    set(validated.recommended_point_ids).issubset(
                        validated.candidate_pool_point_ids
                    )
                )

    def test_schema_rejects_a_point_outside_the_candidate_pool(self):
        payload = next(
            self._build(category)
            for category in RECOMMENDATION_CATEGORIES
            if self._build(category)["has_typical_case"]
        )
        invalid = copy.deepcopy(payload)
        invalid["recommended_point_ids"].append("not-in-candidate-pool")

        with self.assertRaisesRegex(ValueError, "candidate_pool_point_ids"):
            RecommendationPlan.from_dict(invalid)

    def _build(self, category):
        return build_recommendation_plan(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.matrix,
            focus_category=category,
            guidance_metrics=self.metrics,
        )


if __name__ == "__main__":
    unittest.main()
