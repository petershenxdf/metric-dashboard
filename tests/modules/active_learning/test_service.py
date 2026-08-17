import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.modules.active_learning import ActiveLearningService, ActiveLearningStore
from app.modules.active_learning.evidence import CATEGORY_DIMENSIONS
from app.modules.active_learning.schemas import RecommendationPlanV2
from app.modules.active_learning.service import ActiveLearningConflict
from app.modules.active_learning.translation import (
    build_translation_packet,
    translate_plan,
    validate_guidance,
)


def demo_records(count=30):
    records = []
    for index in range(count):
        group = index // 10
        records.append(
            {
                "id": f"record_{index:03d}",
                "x": group * 8 + index % 5,
                "y": group * 3 + index % 3,
                "kind": ("a", "b", "c")[min(group, 2)],
                "truth": f"hidden_{group}",
            }
        )
    return records


def valid_guidance_payload(packet):
    point_ids = list(packet.plan["recommended_point_ids"])
    cards = {
        item["point_id"]: item
        for item in packet.category_evidence_cards
    }
    return {
        "plan_id": packet.plan["plan_id"],
        "category": packet.plan["focus_category"],
        "recommended_point_ids": point_ids,
        "target_rule_ids": list(packet.plan["target_rule_ids"]),
        "label_options": list(packet.label_options),
        "category_explanation": cards[point_ids[0]][
            "category_explanation"
        ],
        "summary": "A summary.",
        "point_guidance": [
            {
                "point_id": point_id,
                "why_selected": "This record is close to the current dividing line.",
                "what_changed_since_last_round": "This is the baseline.",
                "evidence_bullets": [
                    {
                        "dimension_id": bullet["dimension_id"],
                        "status": bullet["status"],
                        "headline": bullet["headline"],
                        "explanation": (
                            bullet["plain_fact"]
                        ),
                        "why_this_point": (
                            "A human label would clarify whether this answer "
                            "matches the record's real-world type."
                        ),
                        "evidence_fact_ids": list(
                            bullet["evidence_fact_ids"]
                        ),
                    }
                    for bullet in cards[point_id]["evidence_bullets"]
                ],
                "comparison_target_ids": [
                    item["point_id"]
                    for item in cards[point_id]["comparison_targets"]
                ],
                "how_to_label": "Compare it with clear examples from both nearby groups.",
                "possible_outcomes": ["The next round can test this label."],
            }
            for point_id in point_ids
        ],
        "warnings": [],
    }


class ActiveLearningServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "active.sqlite3"
        self.store = ActiveLearningStore(self.db_path)
        self.service = ActiveLearningService(self.store)
        prepared = self.service.import_records(
            demo_records(),
            dataset_id="generic_demo",
            entity_name="sample",
            point_id_column="id",
            feature_columns=("x", "y", "kind"),
            ground_truth_columns=("truth",),
        )
        self.session = self.service.create_session(
            prepared.version.dataset_version_id,
            {"n_clusters": 3, "batch_size": 2},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_round_zero_builds_all_categories_and_complete_profiles(self):
        state = self.service.session_state(self.session.session_id)

        self.assertEqual(state["round"]["round_index"], 0)
        self.assertEqual(len(state["round"]["recommendation_plans"]), 8)
        for plan in state["round"]["recommendation_plans"].values():
            candidate_profile_ids = [
                item["point_id"] for item in plan["candidate_point_profiles"]
            ]
            profile_ids = [item["point_id"] for item in plan["point_profiles"]]
            deferred_ids = {
                item["point_id"] for item in plan["deferred_points"]
            }
            self.assertEqual(
                candidate_profile_ids,
                plan["candidate_pool_point_ids"],
            )
            self.assertEqual(profile_ids, plan["recommended_point_ids"])
            self.assertEqual(
                deferred_ids,
                set(plan["candidate_pool_point_ids"])
                - set(plan["recommended_point_ids"]),
            )
            self.assertEqual(
                [item["point_id"] for item in plan["ranking_features"] if item["selected_now"]],
                plan["recommended_point_ids"],
            )
            self.assertEqual(
                set(plan["previous_plan_diff"]["added_reasons"]),
                set(plan["previous_plan_diff"]["added_point_ids"]),
            )
            RecommendationPlanV2.from_dict(plan)

    def test_every_available_category_has_a_fixed_user_facing_evidence_matrix(self):
        state = self.service.session_state(self.session.session_id)
        plans = state["round"]["recommendation_plans"]

        for category, plan in plans.items():
            if not plan["recommended_point_ids"]:
                self.assertEqual(plan["category_evidence_cards"], [])
                continue
            self.assertEqual(
                [item["point_id"] for item in plan["category_evidence_cards"]],
                plan["recommended_point_ids"],
            )
            delegated = plan.get("history_context", {}).get(
                "delegated_category"
            )
            expected_dimensions = list(CATEGORY_DIMENSIONS[category])
            if category == "label_priority" and delegated:
                expected_dimensions.extend(CATEGORY_DIMENSIONS[delegated])
            for card in plan["category_evidence_cards"]:
                self.assertEqual(card["category"], category)
                self.assertEqual(
                    [
                        item["dimension_id"]
                        for item in card["evidence_bullets"]
                    ],
                    expected_dimensions,
                )
                for bullet in card["evidence_bullets"]:
                    self.assertIn(
                        bullet["status"],
                        {"yes", "partly", "no", "insufficient"},
                    )
                    self.assertTrue(bullet["headline"])
                    self.assertTrue(bullet["plain_fact"])
                    self.assertTrue(bullet["why_it_matters"])
                    self.assertTrue(bullet["point_connection"])
                    self.assertTrue(bullet["labeling_value"])
                    self.assertTrue(bullet["evidence_fact_ids"])

    def test_legacy_round_evidence_is_refreshed_without_changing_points(self):
        state = self.service.session_state(self.session.session_id)
        legacy_plan = copy.deepcopy(state["recommendation_plan"])
        legacy_plan["evidence_policy_version"] = "category_evidence_v1"
        for card in legacy_plan["category_evidence_cards"]:
            for bullet in card["evidence_bullets"]:
                bullet.pop("point_connection", None)
                bullet.pop("labeling_value", None)
        round_state = self.store.get_round(state["round"]["round_id"])
        prepared = self.store.load_prepared_dataset(
            self.session.dataset_version_id
        )

        refreshed = self.service._plan_with_current_evidence(
            self.store.get_session(self.session.session_id),
            round_state,
            prepared,
            legacy_plan,
        )

        self.assertEqual(
            refreshed["recommended_point_ids"],
            legacy_plan["recommended_point_ids"],
        )
        self.assertEqual(
            refreshed["evidence_policy_version"],
            "category_evidence_v2",
        )
        self.assertTrue(
            all(
                bullet["point_connection"] and bullet["labeling_value"]
                for card in refreshed["category_evidence_cards"]
                for bullet in card["evidence_bullets"]
            )
        )

    def test_recommendation_schema_rejects_point_contract_changes(self):
        state = self.service.session_state(self.session.session_id)
        payload = dict(state["recommendation_plan"])
        payload["recommended_point_ids"] = ["not_in_candidate_pool"]
        payload["highlighted_point_ids"] = ["not_in_candidate_pool"]

        with self.assertRaisesRegex(ValueError, "candidate pool"):
            RecommendationPlanV2.from_dict(payload)

    def test_same_state_returns_the_same_plan(self):
        first = self.service.session_state(self.session.session_id)
        second = self.service.session_state(self.session.session_id)

        self.assertEqual(
            first["recommendation_plan"]["plan_id"],
            second["recommendation_plan"]["plan_id"],
        )
        self.assertEqual(
            first["recommendation_plan"]["recommended_point_ids"],
            second["recommendation_plan"]["recommended_point_ids"],
        )

    def test_candidate_pool_scales_with_batch_size(self):
        records = [
            {
                "id": f"wide_{index:03d}",
                "x": (index // 30) * 10 + index % 11,
                "y": index % 17,
                "kind": str(index % 4),
            }
            for index in range(90)
        ]
        prepared = self.service.import_records(
            records,
            dataset_id="wide_pool",
            point_id_column="id",
            feature_columns=("x", "y", "kind"),
        )
        session = self.service.create_session(
            prepared.version.dataset_version_id,
            {"n_clusters": 3, "batch_size": 5},
        )
        state = self.service.session_state(session.session_id)

        self.assertEqual(session.config.candidate_pool_size, 15)
        self.assertTrue(
            any(
                plan["candidate_pool_count"] == 15
                for plan in state["round"]["recommendation_plans"].values()
            )
        )

    def test_five_rounds_accumulate_labels_and_recommendation_history(self):
        labeled_points = []
        for round_index in range(5):
            state = self.service.session_state(self.session.session_id)
            plan = state["recommendation_plan"]
            self.assertTrue(plan["recommended_point_ids"])
            point_id = plan["recommended_point_ids"][0]
            selected_row = next(
                item
                for item in plan["candidate_rankings"]
                if item["point_id"] == point_id
            )
            if point_id in labeled_points:
                self.assertTrue(selected_row["recheck_reason"])
            labeled_points.append(point_id)
            self.service.commit_labels(
                self.session.session_id,
                round_id=state["round"]["round_id"],
                expected_label_revision=state["round"]["label_revision"],
                plan_id=plan["plan_id"],
                category=state["focus_category"],
                labels=(
                    {
                        "point_id": point_id,
                        "label_dimension": "semantic_class",
                        "label_value": f"class_{round_index % 2}",
                    },
                ),
            )

        final = self.service.session_state(self.session.session_id)
        self.assertEqual(final["round"]["round_index"], 5)
        self.assertEqual(final["round"]["label_revision"], 5)
        self.assertEqual(len(final["history"]), 6)
        self.assertGreater(len(self.store.all_label_events(self.session.session_id)), 0)
        self.assertTrue(final["round"]["delta"]["recommendation_changes"])

    def test_semantic_label_correction_supersedes_previous_event(self):
        state = self.service.session_state(self.session.session_id)
        point_id = state["recommendation_plan"]["recommended_point_ids"][0]
        self._commit_one(state, point_id, "first")
        next_state = self.service.session_state(self.session.session_id)
        current_plan = next_state["recommendation_plan"]
        source = next_state["focus_category"]
        self.service.commit_labels(
            self.session.session_id,
            round_id=next_state["round"]["round_id"],
            expected_label_revision=next_state["round"]["label_revision"],
            plan_id=current_plan["plan_id"],
            category=source,
            labels=(
                {
                    "point_id": point_id,
                    "label_dimension": "semantic_class",
                    "label_value": "corrected",
                },
            ),
        )

        all_events = [
            event
            for event in self.store.all_label_events(self.session.session_id)
            if event.point_id == point_id
        ]
        self.assertEqual(len(all_events), 2)
        self.assertEqual(
            sorted(event.status for event in all_events),
            ["active", "superseded"],
        )

    def test_sqlite_store_restores_session_after_service_recreation(self):
        state = self.service.session_state(self.session.session_id)
        restored = ActiveLearningService(ActiveLearningStore(self.db_path))
        restored_state = restored.session_state(self.session.session_id)

        self.assertEqual(
            restored_state["round"]["round_id"],
            state["round"]["round_id"],
        )
        self.assertEqual(
            restored_state["recommendation_plan"]["plan_id"],
            state["recommendation_plan"]["plan_id"],
        )

    def test_revert_restores_round_and_active_labels(self):
        baseline = self.service.session_state(self.session.session_id)
        point_id = baseline["recommendation_plan"]["recommended_point_ids"][0]
        self._commit_one(baseline, point_id, "first")

        result = self.service.revert_to_round(
            self.session.session_id,
            baseline["round"]["round_id"],
        )
        restored = self.service.session_state(self.session.session_id)

        self.assertEqual(result["round"]["round_index"], 0)
        self.assertEqual(restored["round"]["round_index"], 0)
        self.assertEqual(restored["active_labels"], [])

    def test_failed_next_round_computation_does_not_partially_commit_labels(self):
        baseline = self.service.session_state(self.session.session_id)
        point_id = baseline["recommendation_plan"]["recommended_point_ids"][0]

        with patch.object(
            self.service,
            "_compute_round",
            side_effect=RuntimeError("analysis failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                self._commit_one(baseline, point_id, "first")

        restored_session = self.store.get_session(self.session.session_id)
        restored_round = self.store.get_round(baseline["round"]["round_id"])
        self.assertEqual(
            restored_session.current_round_id,
            baseline["round"]["round_id"],
        )
        self.assertEqual(restored_session.status, "active")
        self.assertEqual(restored_round.status, "ready_for_labeling")
        self.assertEqual(
            self.store.all_label_events(self.session.session_id),
            (),
        )

    def test_revert_restores_only_labels_from_the_selected_round_ancestry(self):
        baseline = self.service.session_state(self.session.session_id)
        first_point = baseline["recommendation_plan"]["recommended_point_ids"][0]
        self._commit_one(baseline, first_point, "first")

        branch_root = self.service.session_state(self.session.session_id)
        second_point = next(
            point_id
            for point_id in branch_root["recommendation_plan"][
                "recommended_point_ids"
            ]
            if point_id != first_point
        )
        self._commit_one(branch_root, second_point, "second")
        old_branch_head = self.service.session_state(self.session.session_id)

        self.service.revert_to_round(
            self.session.session_id,
            branch_root["round"]["round_id"],
        )
        branch_state = self.service.session_state(self.session.session_id)
        third_point = next(
            point_id
            for point_id in branch_state["dataset_version"]["point_ids"]
            if point_id not in {first_point, second_point}
        )
        self._commit_one(branch_state, third_point, "third")

        self.service.revert_to_round(
            self.session.session_id,
            old_branch_head["round"]["round_id"],
        )
        active_by_point = {
            event.point_id: event.label_value
            for event in self.store.active_label_events(
                self.session.session_id
            )
        }
        self.assertEqual(
            set(active_by_point),
            {first_point, second_point},
        )
        self.assertNotIn(third_point, active_by_point)

    def test_recommendation_history_keeps_computed_shown_selected_and_labeled_separate(self):
        baseline = self.service.session_state(self.session.session_id)
        plan = baseline["recommendation_plan"]
        self.store.record_recommendation_shown(
            session_id=self.session.session_id,
            round_id=baseline["round"]["round_id"],
            plan_id=plan["plan_id"],
            point_ids=plan["recommended_point_ids"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._commit_one(
            baseline,
            plan["recommended_point_ids"][0],
            "first",
        )

        next_state = self.service.session_state(self.session.session_id)
        for next_plan in next_state["round"]["recommendation_plans"].values():
            history = next_plan["history_context"].get(
                "recommendation_history",
                {},
            )
            self.assertEqual(
                set(history),
                {"computed", "shown", "selected", "labeled"},
            )

    def test_label_budget_stops_the_session_without_calling_deepseek(self):
        prepared = self.service.import_records(
            demo_records(36),
            dataset_id="budget_demo",
            entity_name="sample",
            point_id_column="id",
            feature_columns=("x", "y", "kind"),
        )
        session = self.service.create_session(
            prepared.version.dataset_version_id,
            {"n_clusters": 3, "batch_size": 2, "label_budget": 1},
        )
        state = self.service.session_state(session.session_id)
        point_id = state["recommendation_plan"]["recommended_point_ids"][0]
        self.service.commit_labels(
            session.session_id,
            round_id=state["round"]["round_id"],
            expected_round_id=state["round"]["round_id"],
            expected_label_revision=state["round"]["label_revision"],
            plan_id=state["recommendation_plan"]["plan_id"],
            category=state["focus_category"],
            labels=(
                {
                    "point_id": point_id,
                    "label_dimension": "semantic_class",
                    "label_value": "done",
                },
            ),
        )
        stopped = self.service.session_state(session.session_id)
        interpretation = self.service.interpret_category(
            session.session_id,
            round_id=stopped["round"]["round_id"],
            category="label_priority",
            provider_kind="deepseek",
        )

        self.assertEqual(stopped["session"]["status"], "stopped")
        self.assertEqual(stopped["round"]["status"], "stopped")
        self.assertEqual(
            stopped["recommendation_plan"]["stop_reason"],
            "label_budget_reached",
        )
        self.assertEqual(stopped["recommendation_plan"]["recommended_point_ids"], [])
        self.assertTrue(interpretation["diagnostics"]["deepseek_skipped"])
        self.assertIsNone(interpretation["translation_packet"])

    def test_expected_round_id_is_part_of_the_concurrency_contract(self):
        state = self.service.session_state(self.session.session_id)
        plan = state["recommendation_plan"]
        with self.assertRaisesRegex(ActiveLearningConflict, "expected_round_id"):
            self.service.commit_labels(
                self.session.session_id,
                round_id=state["round"]["round_id"],
                expected_round_id="stale_round",
                expected_label_revision=state["round"]["label_revision"],
                plan_id=plan["plan_id"],
                category=state["focus_category"],
                labels=(
                    {
                        "point_id": plan["recommended_point_ids"][0],
                        "label_dimension": "uncertain",
                        "label_value": True,
                    },
                ),
            )

    def test_cross_category_coverage_is_recorded_for_selected_points(self):
        state = self.service.session_state(self.session.session_id)
        for category, plan in state["round"]["recommendation_plans"].items():
            if category == "label_priority":
                continue
            for profile in plan["point_profiles"]:
                self.assertIn(category, profile["covered_categories"])

    def test_translation_packet_is_compact_and_excludes_ground_truth(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name=state["dataset_version"]["entity_name"],
        )
        serialized = str(packet.to_dict())

        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("hidden_", serialized)
        self.assertLess(len(serialized), 15000)
        self.assertEqual(
            packet.plan["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )
        self.assertNotIn(
            "ranking_score_components",
            packet.plan["candidate_rankings"][0],
        )
        self.assertNotIn(
            "selection_reason",
            packet.plan["candidate_rankings"][0],
        )
        self.assertTrue(
            all(
                "plain_language_evidence" in profile
                for profile in packet.point_profiles
            )
        )
        self.assertTrue(packet.category_evidence_cards)
        self.assertNotIn("technical_details", serialized)

    def test_all_category_fallback_guidance_uses_plain_language(self):
        banned_phrases = (
            "affected-region score",
            "candidate rank",
            "category evidence",
            "cluster",
            "cluster lineage",
            "coverage",
            "deterministic",
            "jaccard",
            "outlier score",
            "percentile",
            "purity",
            "ranking score",
            "semantic class",
            "semantic label",
            "ssdbcodi",
            "threshold",
            "unusualness score",
        )
        for category in self.service.session_state(self.session.session_id)[
            "round"
        ]["recommendation_plans"]:
            state = self.service.session_state(
                self.session.session_id,
                focus_category=category,
            )
            guidance = state["guidance"]
            visible_text = " ".join(
                [
                    guidance["category_explanation"],
                    guidance["summary"],
                    *[
                        value
                        for item in guidance["point_guidance"]
                        for value in (
                            item["why_selected"],
                            item["what_changed_since_last_round"],
                            item["how_to_label"],
                            *item["possible_outcomes"],
                            *[
                                bullet["headline"]
                                for bullet in item["evidence_bullets"]
                            ],
                            *[
                                bullet["explanation"]
                                for bullet in item["evidence_bullets"]
                            ],
                            *[
                                bullet["why_this_point"]
                                for bullet in item["evidence_bullets"]
                            ],
                        )
                    ],
                ]
            ).lower()
            for phrase in banned_phrases:
                self.assertNotIn(phrase, visible_text, category)
            self.assertNotRegex(
                visible_text,
                r"\b[a-z][\w.-]*\s*=\s*-?\d+(?:\.\d+)?\b",
                category,
            )

    def test_guidance_validation_rejects_metric_dump_language(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        payload = valid_guidance_payload(packet)
        payload["point_guidance"][0]["why_selected"] = (
            "Category evidence 1.00 and affected-region score 0.88 selected it."
        )

        with self.assertRaisesRegex(ValueError, "technical phrase"):
            validate_guidance(payload, packet)

    def test_valid_plain_language_guidance_passes_validation(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )

        result = validate_guidance(valid_guidance_payload(packet), packet)

        self.assertEqual(
            result["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )
        self.assertTrue(
            all(item["evidence_bullets"] for item in result["point_guidance"])
        )
        for item in result["point_guidance"]:
            for bullet in item["evidence_bullets"]:
                self.assertTrue(bullet["question"])
                self.assertTrue(bullet["why_this_point"])

    def test_guidance_validation_rejects_missing_or_changed_evidence_dimensions(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        missing = valid_guidance_payload(packet)
        missing["point_guidance"][0]["evidence_bullets"].pop()
        changed = valid_guidance_payload(packet)
        changed["point_guidance"][0]["evidence_bullets"][0][
            "status"
        ] = "yes" if (
            changed["point_guidance"][0]["evidence_bullets"][0]["status"]
            != "yes"
        ) else "no"

        with self.assertRaisesRegex(ValueError, "every fixed dimension"):
            validate_guidance(missing, packet)
        with self.assertRaisesRegex(ValueError, "evidence status"):
            validate_guidance(changed, packet)

    def test_guidance_validation_requires_point_specific_labeling_value(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        missing = valid_guidance_payload(packet)
        missing["point_guidance"][0]["evidence_bullets"][0].pop(
            "why_this_point"
        )
        vague = valid_guidance_payload(packet)
        vague["point_guidance"][0]["evidence_bullets"][0][
            "why_this_point"
        ] = "This record matters for the next step."

        with self.assertRaisesRegex(ValueError, "why_this_point"):
            validate_guidance(missing, packet)
        with self.assertRaisesRegex(ValueError, "human label"):
            validate_guidance(vague, packet)

    def test_mock_translation_preserves_deterministic_points(self):
        state = self.service.session_state(self.session.session_id)
        result = self.service.interpret_category(
            self.session.session_id,
            round_id=state["round"]["round_id"],
            category=state["focus_category"],
            provider_kind="mock",
        )

        self.assertEqual(
            result["guidance"]["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )
        self.assertEqual(result["diagnostics"]["provider_kind"], "mock")

    @patch(
        "app.modules.active_learning.translation."
        "DeepSeekClient.generate_json_with_metadata",
        side_effect=RuntimeError("provider unavailable"),
    )
    def test_deepseek_failure_keeps_the_deterministic_round_usable(self, _mock_call):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        result = translate_plan(
            packet,
            provider_kind="deepseek",
            deterministic_fallback=state["guidance"],
        )

        self.assertTrue(result["diagnostics"]["used_fallback"])
        self.assertEqual(
            result["guidance"]["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )

    def test_deepseek_response_from_another_model_is_not_marked_v4_pro(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        response_payload = valid_guidance_payload(packet)
        with patch(
            "app.modules.active_learning.translation."
            "DeepSeekClient.generate_json_with_metadata",
            return_value=(
                response_payload,
                "{}",
                {"model": "deepseek-v4-flash", "usage": {}},
            ),
        ):
            result = translate_plan(
                packet,
                provider_kind="deepseek",
                deterministic_fallback=state["guidance"],
            )

        self.assertTrue(result["diagnostics"]["used_fallback"])
        self.assertFalse(result["diagnostics"]["using_deepseek_v4_pro"])
        self.assertIn("expected deepseek-v4-pro", result["diagnostics"]["error"])

    def test_deepseek_response_without_model_metadata_uses_fallback(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        response_payload = valid_guidance_payload(packet)
        with patch(
            "app.modules.active_learning.translation."
            "DeepSeekClient.generate_json_with_metadata",
            return_value=(
                response_payload,
                "{}",
                {"model": None, "usage": {}},
            ),
        ):
            result = translate_plan(
                packet,
                provider_kind="deepseek",
                deterministic_fallback=state["guidance"],
            )

        self.assertTrue(result["diagnostics"]["used_fallback"])
        self.assertFalse(result["diagnostics"]["using_deepseek_v4_pro"])

    def test_valid_deepseek_v4_pro_translation_is_accepted(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        response_payload = valid_guidance_payload(packet)
        with patch(
            "app.modules.active_learning.translation."
            "DeepSeekClient.generate_json_with_metadata",
            return_value=(
                response_payload,
                "{}",
                {
                    "model": "deepseek-v4-pro",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 80},
                    "finish_reason": "stop",
                    "message_keys": ["role", "content"],
                },
            ),
        ):
            result = translate_plan(
                packet,
                provider_kind="deepseek",
                deterministic_fallback=state["guidance"],
            )

        self.assertTrue(result["diagnostics"]["using_deepseek_v4_pro"])
        self.assertFalse(result["diagnostics"]["used_fallback"])
        self.assertEqual(result["guidance"]["provider_kind"], "deepseek")
        self.assertEqual(
            result["guidance"]["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )

    def test_invalid_deepseek_bullet_uses_local_deterministic_fallback(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        response_payload = valid_guidance_payload(packet)
        response_payload["point_guidance"][0]["evidence_bullets"][0][
            "explanation"
        ] = "Category evidence 1.00 selected it."
        with patch(
            "app.modules.active_learning.translation."
            "DeepSeekClient.generate_json_with_metadata",
            return_value=(
                response_payload,
                "{}",
                {
                    "model": "deepseek-v4-pro",
                    "usage": {},
                    "finish_reason": "stop",
                    "message_keys": ["role", "content"],
                },
            ),
        ):
            result = translate_plan(
                packet,
                provider_kind="deepseek",
                deterministic_fallback=state["guidance"],
            )

        self.assertTrue(result["diagnostics"]["using_deepseek_v4_pro"])
        self.assertTrue(result["diagnostics"]["partial_fallback"])
        self.assertFalse(result["diagnostics"]["used_fallback"])
        repaired = result["guidance"]["point_guidance"][0][
            "evidence_bullets"
        ][0]["explanation"].lower()
        self.assertNotIn("category evidence", repaired)

    def test_guidance_validation_rejects_reordered_points(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        ids = list(packet.plan["recommended_point_ids"])
        payload = {
            "plan_id": packet.plan["plan_id"],
            "category": packet.plan["focus_category"],
            "recommended_point_ids": list(reversed(ids)),
            "category_explanation": "Checks the next labeling question.",
            "summary": "A summary.",
            "point_guidance": [],
            "warnings": [],
        }

        with self.assertRaisesRegex(ValueError, "reordered"):
            validate_guidance(payload, packet)

    def test_guidance_validation_rejects_changed_rules_and_label_options(self):
        state = self.service.session_state(self.session.session_id)
        packet = build_translation_packet(
            plan=state["recommendation_plan"],
            display_rule_set=state["rule_set"],
            round_delta=state["round"]["delta"],
            previous_label_events=(),
            entity_name="sample",
        )
        changed_rules = valid_guidance_payload(packet)
        changed_rules["target_rule_ids"] = ["invented_rule"]
        changed_labels = valid_guidance_payload(packet)
        changed_labels["label_options"] = ["invented_label"]

        with self.assertRaisesRegex(ValueError, "target rule"):
            validate_guidance(changed_rules, packet)
        with self.assertRaisesRegex(ValueError, "label options"):
            validate_guidance(changed_labels, packet)

    def _commit_one(self, state, point_id, label):
        plan = state["recommendation_plan"]
        return self.service.commit_labels(
            self.session.session_id,
            round_id=state["round"]["round_id"],
            expected_label_revision=state["round"]["label_revision"],
            plan_id=plan["plan_id"],
            category=state["focus_category"],
            labels=(
                {
                    "point_id": point_id,
                    "label_dimension": "semantic_class",
                    "label_value": label,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
