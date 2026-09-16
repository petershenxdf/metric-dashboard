import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.modules.active_learning import (
    ActiveLearningService,
    ActiveLearningStore,
    SessionConfig,
)
from app.modules.active_learning.review import CATEGORY_IDS
from app.modules.active_learning.service import ActiveLearningConflict
from app.modules.algorithm_adapters.service import run_default_analysis


def demo_records(count=30):
    return [
        {
            "id": f"record_{i:03d}",
            "x": (i // 10) * 8 + i % 5,
            "y": (i // 10) * 3 + i % 3,
            "kind": "a" if i < 15 else "b",
            "truth": "PRIVATE_TRUTH",
        }
        for i in range(count)
    ]


class ActiveLearningServiceTests(unittest.TestCase):
    def test_connections_close_after_commit_and_rollback(self):
        with self.store._connect() as connection:
            connection.execute("CREATE TABLE connection_check (value INTEGER)")
            connection.execute("INSERT INTO connection_check VALUES (1)")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with self.store._connect() as failed_connection:
                failed_connection.execute("INSERT INTO connection_check VALUES (2)")
                raise RuntimeError("abort")
        with self.assertRaises(sqlite3.ProgrammingError):
            failed_connection.execute("SELECT 1")
        with self.store._connect() as fresh:
            self.assertEqual(fresh.execute("SELECT COUNT(*) FROM connection_check").fetchone()[0], 1)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "test.sqlite3"
        self.store = ActiveLearningStore(self.path)
        self.service = ActiveLearningService(self.store)
        self.prepared = self.service.import_records(
            demo_records(), point_id_column="id", ground_truth_columns=("truth",)
        )
        self.session = self.service.create_session(
            self.prepared.version.dataset_version_id
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def state(self):
        return self.service.session_state(self.session.session_id)

    def commit(
        self,
        pid="record_000",
        dimension="semantic_class",
        value="human type",
        state=None,
    ):
        state = state or self.state()
        return self.service.commit_labels(
            self.session.session_id,
            round_id=state["round"]["round_id"],
            expected_round_id=state["round"]["round_id"],
            expected_label_revision=state["round"]["label_revision"],
            category="local_sparsity",
            labels=[
                {"point_id": pid, "label_dimension": dimension, "label_value": value}
            ],
        )

    def test_all_outlier_commit_and_normal_correction_are_persisted_and_reversible(self):
        initial = self.state()
        self.service.commit_labels(
            self.session.session_id,
            round_id=initial["round"]["round_id"],
            expected_round_id=initial["round"]["round_id"],
            expected_label_revision=initial["round"]["label_revision"],
            category="local_sparsity",
            labels=[{"point_id": pid, "label_dimension": "outlier_status", "label_value": True}
                    for pid in self.prepared.version.point_ids],
        )
        all_outliers = self.state()
        self.assertEqual(len(all_outliers["active_labels"]), 30)
        self.assertTrue(all(p["is_outlier"] for p in all_outliers["plot_points"]))
        self.assertEqual(all_outliers["review"]["instability_settings"], [2, 3, 4])
        self.assertTrue(all(c["pool_size"] == 0 for c in all_outliers["review"]["categories"]))
        self.commit("record_000", "outlier_status", False)
        corrected = self.state()
        normal = [p for p in corrected["plot_points"] if not p["is_outlier"]]
        self.assertEqual([p["point_id"] for p in normal], ["record_000"])
        self.service.revert_to_round(self.session.session_id, initial["round"]["round_id"])
        self.assertEqual(self.state()["review"], initial["review"])
        self.assertEqual(self.state()["active_labels"], [])

    def test_six_scores_and_no_old_contracts_or_ground_truth(self):
        state = self.state()
        self.assertEqual(
            [c["id"] for c in state["review"]["categories"]], list(CATEGORY_IDS)
        )
        self.assertEqual(len(state["review"]["rows"]), 30)
        self.assertNotIn("recommendation_plans", state["round"])
        self.assertNotIn("recommendation_plan", state)
        self.assertNotIn("PRIVATE_TRUTH", json.dumps(state))
        self.assertEqual(state["review"]["instability_settings"], [2, 3, 4])
        for row in state["review"]["rows"]:
            self.assertEqual(set(row["scores"]), set(CATEGORY_IDS))
        self.assertFalse(hasattr(self.service, "interpret_category"))

    def test_reruns_are_real_full_pipeline_with_fixed_inputs(self):
        with patch(
            "app.modules.active_learning.service.run_default_analysis",
            wraps=run_default_analysis,
        ) as run:
            self.commit()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            sorted(call.kwargs["provider"]._min_pts for call in run.call_args_list),
            [2, 3, 4],
        )
        for call in run.call_args_list:
            self.assertEqual(call.args[0], self.prepared.feature_matrix)
            self.assertEqual(call.kwargs["n_clusters"], 3)
            self.assertEqual(
                len(call.kwargs["provider"]._labeling_state.annotations), 1
            )

    def test_commit_refreshes_snapshot_and_reuses_projection(self):
        before = self.state()
        result = self.commit()
        after = self.state()
        self.assertEqual(after["round"]["round_index"], 1)
        self.assertEqual(after["round"]["label_revision"], 1)
        self.assertNotEqual(before["round"]["round_id"], after["round"]["round_id"])
        self.assertEqual(before["round"]["projection"], after["round"]["projection"])
        self.assertEqual(len(result["events"]), 1)
        self.assertTrue(
            any(
                c["pool_size"]
                for c in after["review"]["categories"]
                if c["id"] == "label_coverage_gap"
            )
        )
        labeled = next(
            row for row in after["review"]["rows"] if row["point_id"] == "record_000"
        )
        self.assertEqual(labeled["review_status"], "labeled")
        self.assertFalse(labeled["eligible"])

    def test_stale_round_and_revision_do_not_commit(self):
        stale = self.state()
        self.commit()
        with self.assertRaises(ActiveLearningConflict):
            self.commit(state=stale)
        current = self.state()
        current["round"]["label_revision"] = 999
        with self.assertRaises(ActiveLearningConflict):
            self.commit(state=current)
        self.assertEqual(
            len(self.store.active_label_events(self.session.session_id)), 1
        )

    def test_failure_before_persistence_is_atomic(self):
        before = self.state()
        with patch(
            "app.modules.active_learning.service.run_default_analysis",
            side_effect=RuntimeError("failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.commit()
        self.assertEqual(before, self.state())
        self.assertEqual(len(self.store.all_label_events(self.session.session_id)), 0)

    def test_correction_supersedes_same_dimension(self):
        self.commit(value="type A")
        self.commit(value="type B")
        events = self.store.all_label_events(self.session.session_id)
        self.assertEqual(sum(e.status == "active" for e in events), 1)
        self.assertEqual(sum(e.status == "superseded" for e in events), 1)
        self.assertTrue(
            next(e for e in events if e.status == "active").supersedes_event_id
        )

    def test_unsure_never_becomes_reference_and_defers_one_round(self):
        self.commit(dimension="uncertain", value=True)
        review = self.state()["review"]
        point = next(row for row in review["rows"] if row["point_id"] == "record_000")
        self.assertEqual(point["review_status"], "deferred")
        for category in (
            "label_coverage_gap",
            "weak_group_reachability",
            "known_outlier_similarity",
        ):
            self.assertEqual(
                next(c for c in review["categories"] if c["id"] == category)[
                    "pool_size"
                ],
                0,
            )
        self.commit(pid="record_001", dimension="uncertain", value=True)
        point = next(
            row
            for row in self.state()["review"]["rows"]
            if row["point_id"] == "record_000"
        )
        self.assertEqual(point["review_status"], "unlabeled")

    def test_revert_and_new_branch_keep_only_ancestral_labels(self):
        root = self.state()["round"]["round_id"]
        first = self.commit(value="type A")["round"]["round_id"]
        old_branch = self.commit(pid="record_001")["round"]["round_id"]
        self.service.revert_to_round(self.session.session_id, first)
        self.assertEqual(
            len(self.store.active_label_events(self.session.session_id)), 1
        )
        new_branch = self.commit(pid="record_002")["round"]["round_id"]
        self.assertNotEqual(old_branch, new_branch)
        self.service.revert_to_round(self.session.session_id, root)
        self.assertEqual(self.store.active_label_events(self.session.session_id), ())
        self.assertEqual(len(self.store.all_label_events(self.session.session_id)), 3)

    def test_restart_and_get_do_not_rerun_or_change_scores(self):
        self.commit()
        before = self.state()
        self.service = ActiveLearningService(ActiveLearningStore(self.path))
        with patch(
            "app.modules.active_learning.service.run_default_analysis",
            side_effect=AssertionError("GET must not run analysis"),
        ):
            after = self.state()
        self.assertEqual(before, after)

    def test_legacy_round_requires_explicit_non_destructive_upgrade(self):
        current = self.store.get_round(self.session.current_round_id)
        self.store.save_round(replace(current, review={}))
        self.assertTrue(self.state()["legacy_round"])
        result = self.service.upgrade_round(self.session.session_id, current.round_id)
        self.assertEqual(result["round"]["parent_round_id"], current.round_id)
        self.assertEqual(result["round"]["label_revision"], 0)
        self.assertTrue(self.state()["review"])
        self.assertEqual(
            self.store.get_round(current.round_id).analysis, current.analysis
        )
        with self.assertRaises(ValueError):
            self.service.upgrade_round(
                self.session.session_id, result["round"]["round_id"]
            )

    def test_stopped_legacy_round_can_upgrade(self):
        current = self.store.get_round(self.session.current_round_id)
        self.store.save_round(replace(current, review={}, status="stopped"))
        self.store.save_session(replace(self.session, status="stopped"))
        self.service.upgrade_round(self.session.session_id, current.round_id)
        self.assertEqual(self.state()["session"]["status"], "active")

    def test_budget_stops_but_preserves_score_values(self):
        self.session = self.service.create_session(
            self.prepared.version.dataset_version_id, {"label_budget": 1}
        )
        self.commit()
        state = self.state()
        self.assertEqual(state["session"]["status"], "stopped")
        self.assertTrue(
            any(
                row["scores"]["local_sparsity"]["raw"] is not None
                for row in state["review"]["rows"]
            )
        )
        self.assertFalse(
            any(c["recommended_point_ids"] for c in state["review"]["categories"])
        )

    def test_no_useful_recommendation_does_not_block_manual_review(self):
        self.session = self.service.create_session(
            self.prepared.version.dataset_version_id,
            {"n_clusters": 1, "review_percentile_threshold": 100},
        )
        self.assertEqual(self.state()["session"]["status"], "active")
        self.assertFalse(
            any(
                c["recommended_point_ids"] for c in self.state()["review"]["categories"]
            )
        )
        self.commit()

    def test_settings_validate_and_boundary_instability_runs(self):
        for config in (
            {"min_pts": 0},
            {"min_pts": 30},
            {"review_percentile_threshold": 101},
        ):
            with self.assertRaises(ValueError):
                self.service.create_session(
                    self.prepared.version.dataset_version_id, config
                )
        self.session = self.service.create_session(
            self.prepared.version.dataset_version_id, {"min_pts": 1}
        )
        self.assertEqual(self.state()["review"]["instability_settings"], [1, 2])
        self.assertEqual(SessionConfig.from_dict({}).min_pts, 3)

    def test_reject_duplicate_batch_dimensions_and_bad_payloads(self):
        state = self.state()
        for labels in (
            [None],
            "labels",
            [],
            [{"point_id": "missing"}],
            [
                {
                    "point_id": "record_000",
                    "label_dimension": "outlier_status",
                    "label_value": True,
                }
            ]
            * 2,
        ):
            with self.assertRaises(ValueError):
                self.service.commit_labels(
                    self.session.session_id,
                    round_id=state["round"]["round_id"],
                    expected_label_revision=0,
                    labels=labels,
                )

    def test_ground_truth_feature_and_metadata_overlap_rejected(self):
        for kwargs in (
            {"feature_columns": ("x", "truth")},
            {"metadata_columns": ("truth",)},
        ):
            with self.assertRaises(ValueError):
                self.service.import_records(
                    demo_records(), ground_truth_columns=("truth",), **kwargs
                )

    def test_ground_truth_changes_do_not_change_scores(self):
        records = demo_records()
        for record in records:
            record["truth"] = "different hidden truth"
        prepared = self.service.import_records(
            records, point_id_column="id", ground_truth_columns=("truth",)
        )
        session = self.service.create_session(prepared.version.dataset_version_id)
        self.assertEqual(
            self.state()["review"],
            self.service.session_state(session.session_id)["review"],
        )

    def test_actual_legacy_round_json_loads_without_new_field(self):
        old = self.store.get_round(self.session.current_round_id).to_dict()
        del old["review"]
        old["recommendation_plans"] = {"label_priority": {"plan_id": "legacy_plan"}}
        with self.store._connect() as connection:
            connection.execute(
                "UPDATE active_learning_rounds SET payload_json = ? WHERE round_id = ?",
                (json.dumps(old), old["round_id"]),
            )
        self.assertTrue(self.state()["legacy_round"])
        self.service.upgrade_round(self.session.session_id, old["round_id"])
        self.assertTrue(self.state()["review"])

    def test_stale_revert_does_not_change_labels_or_head(self):
        root = self.state()["round"]["round_id"]
        self.commit()
        before = self.state()
        with self.assertRaisesRegex(ValueError, "session changed"):
            self.store.revert_events_to_round(
                self.session.session_id, root, expected_round_id=root, updated_at="now"
            )
        self.assertEqual(self.state(), before)


if __name__ == "__main__":
    unittest.main()
