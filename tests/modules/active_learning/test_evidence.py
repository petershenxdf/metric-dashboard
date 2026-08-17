import json
import tempfile
import unittest
from pathlib import Path

from app.modules.active_learning import ActiveLearningService, ActiveLearningStore
from app.modules.active_learning.evidence import (
    CATEGORY_DIMENSIONS,
    EVIDENCE_POLICY_VERSION,
    build_category_evidence_cards,
)
from app.modules.active_learning.service import (
    _projection_from_dict,
    analysis_from_dict,
    rule_set_from_dict,
)


class CategoryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ActiveLearningStore(
            Path(self.tempdir.name) / "evidence.sqlite3"
        )
        self.service = ActiveLearningService(self.store)
        records = [
            {
                "id": f"item_{index:02d}",
                "length": (index // 8) * 7 + index % 4,
                "weight": (index // 8) * 2 + index % 3,
                "kind": ("a", "b", "c")[min(index // 8, 2)],
                "private_truth": f"hidden_{index // 8}",
            }
            for index in range(24)
        ]
        prepared = self.service.import_records(
            records,
            dataset_id="evidence_fixture",
            point_id_column="id",
            feature_columns=("length", "weight", "kind"),
            ground_truth_columns=("private_truth",),
        )
        self.session = self.service.create_session(
            prepared.version.dataset_version_id,
            {"n_clusters": 3, "batch_size": 3},
        )
        self.prepared = self.store.load_prepared_dataset(
            prepared.version.dataset_version_id
        )
        self.state = self.service.session_state(self.session.session_id)
        self.analysis = analysis_from_dict(self.state["analysis"])
        self.rule_set = rule_set_from_dict(self.state["model_rule_set"])
        self.projection = _projection_from_dict(
            self.state["round"]["projection"]
        )
        self.point_id = self.state["recommendation_plan"][
            "recommended_point_ids"
        ][0]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_all_categories_produce_their_fixed_user_facing_dimensions(self):
        for category, expected in CATEGORY_DIMENSIONS.items():
            delegated = (
                "boundary_review" if category == "label_priority" else ""
            )
            plan = self._plan(category, delegated=delegated)

            cards = build_category_evidence_cards(
                plan,
                prepared=self.prepared,
                analysis=self.analysis,
                rule_set=self.rule_set,
                projection=self.projection,
                active_events=(),
                parent_round=None,
                label_vocabulary={},
            )

            self.assertEqual(len(cards), 1)
            card = cards[0]
            expected_dimensions = list(expected)
            if delegated:
                expected_dimensions.extend(CATEGORY_DIMENSIONS[delegated])
            self.assertEqual(
                [
                    bullet["dimension_id"]
                    for bullet in card["evidence_bullets"]
                ],
                expected_dimensions,
            )
            self.assertEqual(
                card["evidence_policy_version"],
                EVIDENCE_POLICY_VERSION,
            )
            for bullet in card["evidence_bullets"]:
                self.assertTrue(bullet["question"])
                self.assertTrue(bullet["headline"])
                self.assertTrue(bullet["plain_fact"])
                self.assertTrue(bullet["why_it_matters"])
                self.assertTrue(bullet["point_connection"])
                self.assertTrue(bullet["labeling_value"])
                self.assertTrue(bullet["evidence_fact_ids"])

    def test_evidence_is_stable_and_never_contains_ground_truth(self):
        plan = self._plan("boundary_review")
        first = build_category_evidence_cards(
            plan,
            prepared=self.prepared,
            analysis=self.analysis,
            rule_set=self.rule_set,
            projection=self.projection,
            active_events=(),
            parent_round=None,
            label_vocabulary={},
        )
        second = build_category_evidence_cards(
            plan,
            prepared=self.prepared,
            analysis=self.analysis,
            rule_set=self.rule_set,
            projection=self.projection,
            active_events=(),
            parent_round=None,
            label_vocabulary={},
        )
        serialized = json.dumps(first, sort_keys=True)

        self.assertEqual(first, second)
        self.assertNotIn("private_truth", serialized)
        self.assertNotIn("hidden_", serialized)
        neighborhood = next(
            bullet
            for bullet in first[0]["evidence_bullets"]
            if bullet["dimension_id"] == "boundary_other_group_nearby"
        )
        self.assertEqual(
            neighborhood["technical_details"]["computed_in"],
            "preprocessed_full_feature_space",
        )

    def test_missing_human_labels_are_reported_as_insufficient(self):
        expected_insufficient = {
            "overlap_merge_signal": "overlap_human_labels",
            "split_or_new_cluster_signal": "split_human_labels",
            "anomaly_label_review": "anomaly_confirmed_examples",
            "feature_label_strategy": "feature_human_label_pattern",
            "rule_confidence_audit": "audit_human_label_agreement",
        }
        for category, dimension_id in expected_insufficient.items():
            cards = build_category_evidence_cards(
                self._plan(category),
                prepared=self.prepared,
                analysis=self.analysis,
                rule_set=self.rule_set,
                projection=self.projection,
                active_events=(),
                parent_round=None,
                label_vocabulary={},
            )
            bullet = next(
                item
                for item in cards[0]["evidence_bullets"]
                if item["dimension_id"] == dimension_id
            )
            self.assertEqual(bullet["status"], "insufficient", category)
            self.assertIn("human", bullet["plain_fact"].lower())

    def test_rule_audit_connects_each_rule_finding_to_the_selected_record(self):
        cards = build_category_evidence_cards(
            self._plan("rule_confidence_audit"),
            prepared=self.prepared,
            analysis=self.analysis,
            rule_set=self.rule_set,
            projection=self.projection,
            active_events=(),
            parent_round=None,
            label_vocabulary={},
        )

        for bullet in cards[0]["evidence_bullets"]:
            self.assertIn("record", bullet["point_connection"].lower())
            self.assertTrue(
                any(
                    cue in bullet["labeling_value"].lower()
                    for cue in ("label", "human", "type")
                )
            )

    def _plan(self, category, *, delegated=""):
        rule_ids = [
            rule["rule_id"]
            for rule in self.state["model_rule_set"]["rules"]
        ]
        profile = {
            "point_id": self.point_id,
            "covered_categories": [category],
        }
        history_context = (
            {"delegated_category": delegated} if delegated else {}
        )
        return {
            "plan_id": f"test_{category}",
            "focus_category": category,
            "recommended_point_ids": [self.point_id],
            "target_rule_ids": rule_ids,
            "point_profiles": [profile],
            "candidate_rankings": [
                {
                    "point_id": self.point_id,
                    "candidate_rank": 1,
                    "selection_reason": "fixed test selection",
                    "recheck_reason": "",
                }
            ],
            "previous_plan_diff": {"added_point_ids": [self.point_id]},
            "history_context": history_context,
        }


if __name__ == "__main__":
    unittest.main()
