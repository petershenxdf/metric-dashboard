import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app


def records():
    return [
        {
            "id": f"p{index:02d}",
            "x": (index // 8) * 6 + index % 4,
            "y": (index // 8) * 2 + index % 3,
            "kind": "left" if index < 12 else "right",
            "truth": "private",
        }
        for index in range(24)
    ]


class ActiveLearningDashboardWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "active.sqlite3")
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["ACTIVE_LEARNING_DB_PATH"] = self.db_path
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def create_session(self):
        dataset = self.client.post(
            "/api/datasets",
            json={
                "dataset_id": "workflow_demo",
                "entity_name": "item",
                "records": records(),
                "point_id_column": "id",
                "feature_columns": ["x", "y", "kind"],
                "ground_truth_columns": ["truth"],
            },
        )
        dataset_version_id = dataset.json["data"]["dataset_version_id"]
        session = self.client.post(
            "/api/active-learning/sessions",
            json={
                "dataset_version_id": dataset_version_id,
                "config": {"n_clusters": 3, "batch_size": 3},
            },
        )
        return session.json["data"]

    def test_index_and_generic_session_page_load(self):
        self.assertEqual(
            self.client.get("/workflows/active-learning-dashboard/").status_code,
            200,
        )
        created = self.create_session()
        session_id = created["session"]["session_id"]
        page = self.client.get(
            f"/workflows/active-learning-dashboard/{session_id}/"
        )

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Generic Active Learning Dashboard", self.client.get("/workflows/active-learning-dashboard/").data)
        self.assertIn(b"Commit The Next Batch", page.data)
        self.assertIn(b"Learning Rounds", page.data)
        self.assertIn(b"Explain With DeepSeek V4 Pro", page.data)
        self.assertNotIn(b"encoded::", page.data)
        self.assertNotIn(b"private", page.data)
        self.assertNotIn(b"wine sample", page.data.lower())
        self.assertIn(b"Why this category", page.data)
        self.assertIn(b"Why this record was recommended", page.data)
        self.assertIn(b'class="evidence-question"', page.data)
        self.assertIn(b"What we see:", page.data)
        self.assertIn(b"Why label this record:", page.data)
        self.assertIn(b"How to label it", page.data)
        self.assertIn(b"What your answer would tell us", page.data)
        self.assertIn(b"Technical details", page.data)
        self.assertIn(b"Compare with", page.data)
        self.assertIn(b"data-comparison-point-id", page.data)
        self.assertIn(b"data-feature-name", page.data)
        self.assertIn(b"data-rule-feature", page.data)
        self.assertIn(b'evidence-status', page.data)
        self.assertNotIn(b">Observed<", page.data)
        self.assertIn(b"data-generate-explanation", page.data)
        self.assertIn(b"data-guide-point-id", page.data)
        self.assertIn(b"data-clear-guidance-focus", page.data)
        self.assertIn(b"guidance-callout-line", page.data)
        self.assertNotIn(b"category evidence", page.data.lower())
        self.assertNotIn(b"affected-region score", page.data.lower())
        self.assertNotIn(b"SSDBCODI seed", page.data)
        self.assertNotIn(b"generate=1", page.data)

    def test_state_exposes_round_plan_delta_and_no_ground_truth(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        response = self.client.get(
            f"/api/active-learning/sessions/{session_id}/state"
        )

        self.assertEqual(response.status_code, 200)
        state = response.json["data"]
        self.assertEqual(state["round"]["round_index"], 0)
        self.assertIn("recommendation_plan", state)
        self.assertIn("candidate_rankings", state["recommendation_plan"])
        self.assertIn(
            "category_evidence_cards",
            state["recommendation_plan"],
        )
        self.assertTrue(
            state["recommendation_plan"]["category_evidence_cards"]
        )
        self.assertIn("delta", state["round"])
        self.assertEqual(len(state["round"]["recommendation_plans"]), 8)
        self.assertTrue(
            all(
                plan["evidence_policy_version"]
                == "category_evidence_v2"
                for plan in state["round"][
                    "recommendation_plans"
                ].values()
            )
        )
        self.assertNotIn("ground_truth", str(state["plot_points"]))

    def test_label_commit_advances_round_and_stale_commit_is_rejected(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        state = created["state"]
        plan = state["recommendation_plan"]
        point_id = plan["recommended_point_ids"][0]
        endpoint = (
            f"/api/active-learning/sessions/{session_id}/rounds/"
            f"{state['round']['round_id']}/labels"
        )
        payload = {
            "expected_round_id": state["round"]["round_id"],
            "expected_label_revision": state["round"]["label_revision"],
            "plan_id": plan["plan_id"],
            "category": state["focus_category"],
            "labels": [
                {
                    "point_id": point_id,
                    "label_dimension": "semantic_class",
                    "label_value": "domain_a",
                }
            ],
        }
        committed = self.client.post(endpoint, json=payload)
        stale = self.client.post(endpoint, json=payload)

        self.assertEqual(committed.status_code, 200)
        self.assertEqual(
            committed.json["data"]["round"]["round_index"],
            1,
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json["error"]["code"], "stale_round")

    def test_mismatched_expected_round_id_is_rejected(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        state = created["state"]
        plan = state["recommendation_plan"]
        response = self.client.post(
            (
                f"/api/active-learning/sessions/{session_id}/rounds/"
                f"{state['round']['round_id']}/labels"
            ),
            json={
                "expected_round_id": "older_round",
                "expected_label_revision": state["round"]["label_revision"],
                "plan_id": plan["plan_id"],
                "category": state["focus_category"],
                "labels": [
                    {
                        "point_id": plan["recommended_point_ids"][0],
                        "label_dimension": "uncertain",
                        "label_value": True,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json["error"]["code"], "stale_round")

    def test_mock_interpretation_is_cached_and_keeps_plan_ids(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        state = created["state"]
        category = state["focus_category"]
        endpoint = (
            f"/api/active-learning/sessions/{session_id}/rounds/"
            f"{state['round']['round_id']}/categories/{category}/interpret"
        )
        first = self.client.post(endpoint, json={"provider_kind": "mock"})
        second = self.client.post(endpoint, json={"provider_kind": "mock"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(
            first.json["data"]["guidance"]["recommended_point_ids"],
            state["recommendation_plan"]["recommended_point_ids"],
        )
        self.assertTrue(second.json["data"]["cache_hit"])

    @patch(
        "app.modules.active_learning.service."
        "ActiveLearningService.interpret_category"
    )
    def test_dashboard_get_never_triggers_an_interpretation_call(
        self,
        interpret,
    ):
        created = self.create_session()
        session_id = created["session"]["session_id"]

        response = self.client.get(
            f"/workflows/active-learning-dashboard/{session_id}/"
            "?provider_kind=deepseek&generate=1"
        )

        self.assertEqual(response.status_code, 200)
        interpret.assert_not_called()

    def test_dashboard_records_each_shown_recommendation_only_once(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        path = f"/workflows/active-learning-dashboard/{session_id}/"

        self.client.get(path)
        self.client.get(path)

        with sqlite3.connect(self.db_path) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM active_learning_recommendation_events
                WHERE session_id = ? AND event_kind = 'shown'
                """,
                (session_id,),
            ).fetchone()[0]
        self.assertEqual(
            count,
            len(created["state"]["recommendation_plan"]["recommended_point_ids"]),
        )

    def test_dashboard_ignores_interpretation_from_an_old_prompt_contract(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        state = created["state"]
        category = state["focus_category"]
        endpoint = (
            f"/api/active-learning/sessions/{session_id}/rounds/"
            f"{state['round']['round_id']}/categories/{category}/interpret"
        )
        self.client.post(endpoint, json={"provider_kind": "mock"})
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE active_learning_interpretations
                SET diagnostics_json = ?
                WHERE round_id = ? AND plan_id = ? AND provider_kind = 'mock'
                """,
                (
                    json.dumps({"prompt_template_version": "old_contract"}),
                    state["round"]["round_id"],
                    state["recommendation_plan"]["plan_id"],
                ),
            )

        page = self.client.get(
            f"/workflows/active-learning-dashboard/{session_id}/"
            "?provider_kind=mock&show_interpretation=1"
        )

        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b"cache hit", page.data)

    def test_session_survives_app_recreation(self):
        created = self.create_session()
        session_id = created["session"]["session_id"]
        second_app = create_app()
        second_app.config["TESTING"] = True
        second_app.config["ACTIVE_LEARNING_DB_PATH"] = self.db_path
        response = second_app.test_client().get(
            f"/api/active-learning/sessions/{session_id}/state"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["data"]["session"]["session_id"],
            session_id,
        )


if __name__ == "__main__":
    unittest.main()
