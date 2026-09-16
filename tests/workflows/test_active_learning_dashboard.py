import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.modules.active_learning import ActiveLearningStore


def records():
    return [
        {
            "id": f"p{i:02d}",
            "x": (i // 8) * 6 + i % 4,
            "y": i % 3,
            "truth": "PRIVATE_GROUND_TRUTH",
        }
        for i in range(24)
    ]


class ActiveLearningDashboardWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.app = create_app()
        self.app.config.update(
            TESTING=True,
            ACTIVE_LEARNING_DB_PATH=str(Path(self.tempdir.name) / "test.sqlite3"),
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def create_session(self):
        dataset = self.client.post(
            "/api/datasets",
            json={
                "records": records(),
                "point_id_column": "id",
                "ground_truth_columns": ["truth"],
            },
        )
        self.assertEqual(dataset.status_code, 201)
        version = dataset.get_json()["data"]["dataset_version_id"]
        response = self.client.post(
            "/api/active-learning/sessions",
            json={"dataset_version_id": version, "config": {"n_clusters": 3}},
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["data"]["state"]

    def label_url(self, state):
        return (
            "/api/active-learning/sessions/"
            + state["session"]["session_id"]
            + "/rounds/"
            + state["round"]["round_id"]
            + "/labels"
        )

    def payload(self, state):
        return {
            "expected_round_id": state["round"]["round_id"],
            "expected_label_revision": state["round"]["label_revision"],
            "category": "local_sparsity",
            "labels": [
                {
                    "point_id": "p00",
                    "label_dimension": "outlier_status",
                    "label_value": True,
                }
            ],
        }

    def test_index_and_six_column_dashboard(self):
        self.assertEqual(
            self.client.get("/workflows/active-learning-dashboard/").status_code, 200
        )
        state = self.create_session()
        response = self.client.get(
            "/workflows/active-learning-dashboard/"
            + state["session"]["session_id"]
            + "/"
        )
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for text in (
            "Review priority matrix",
            "Match ALL",
            "Match ANY",
            "No percentile (NA / Reviewed / Deferred)",
            'id="label-scope"',
            'id="review-guidance"',
            'class="batch-ring"',
            "Normal",
            "Unsure",
            "Model Instability",
        ):
            self.assertIn(text, html)
        self.assertNotIn("DeepSeek", html)
        self.assertNotIn("PRIVATE_GROUND_TRUTH", html)
        self.assertEqual(html.count("data-category-heading="), 6)

    def test_labels_need_round_and_revision_but_not_old_plan(self):
        state = self.create_session()
        response = self.client.post(self.label_url(state), json=self.payload(state))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"]["round"]["round_index"], 1)
        stale = self.client.post(self.label_url(state), json=self.payload(state))
        self.assertEqual(stale.status_code, 409)

    def test_old_interpret_endpoint_removed(self):
        state = self.create_session()
        response = self.client.post(
            self.label_url(state).replace(
                "/labels", "/categories/label_priority/interpret"
            ),
            json={"provider_kind": "deepseek"},
        )
        self.assertEqual(response.status_code, 404)

    def test_get_never_runs_model_or_network(self):
        state = self.create_session()
        sid = state["session"]["session_id"]
        with (
            patch(
                "app.modules.active_learning.service.run_default_analysis",
                side_effect=AssertionError("unexpected rerun"),
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("unexpected network"),
            ),
        ):
            response = self.client.get(
                "/api/active-learning/sessions/"
                + sid
                + "/state?focus_category=label_priority"
            )
            page = self.client.get(
                "/workflows/active-learning-dashboard/"
                + sid
                + "/?show_interpretation=1"
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(page.status_code, 200)
        self.assertEqual(response.get_json()["data"]["review"], state["review"])

    def test_history_revert_and_restart(self):
        state = self.create_session()
        self.client.post(self.label_url(state), json=self.payload(state))
        sid = state["session"]["session_id"]
        history = self.client.get(
            "/api/active-learning/sessions/" + sid + "/history"
        ).get_json()["data"]
        self.assertEqual(len(history["rounds"]), 2)
        self.assertEqual(len(history["label_events"]), 1)
        response = self.client.post(self.label_url(state).replace("/labels", "/revert"))
        self.assertEqual(response.status_code, 200)
        app = create_app()
        app.config.update(self.app.config)
        resumed = (
            app.test_client()
            .get("/api/active-learning/sessions/" + sid + "/state")
            .get_json()["data"]
        )
        self.assertEqual(resumed["review"], state["review"])
        self.assertEqual(resumed["active_labels"], [])

    def test_invalid_labels_return_structured_errors(self):
        state = self.create_session()
        for payload in (
            ["not an object"],
            {},
            {
                "expected_round_id": state["round"]["round_id"],
                "expected_label_revision": "not-int",
            },
            {**self.payload(state), "category": "label_priority"},
            {**self.payload(state), "labels": [None]},
        ):
            response = self.client.post(self.label_url(state), json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.get_json()["ok"])

    def test_unknown_session_returns_404(self):
        response = self.client.get("/api/active-learning/sessions/missing/state")
        self.assertEqual(response.status_code, 404)

    def test_static_matrix_resources_available(self):
        for path in ("/static/review_matrix.js", "/static/review_matrix.css"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_wine_fixture_uses_the_same_six_score_workflow(self):
        response = self.client.post(
            "/workflows/active-learning-dashboard/wine-fixture", follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("wine_mat", response.get_data(as_text=True))
        self.assertEqual(response.get_data(as_text=True).count("data-category-heading="), 6)

    def test_legacy_dashboard_and_explicit_upgrade_endpoint(self):
        state = self.create_session()
        store = ActiveLearningStore(self.app.config["ACTIVE_LEARNING_DB_PATH"])
        old = store.get_round(state["round"]["round_id"])
        store.save_round(replace(old, review={}))
        sid = state["session"]["session_id"]
        page = self.client.get("/workflows/active-learning-dashboard/" + sid + "/")
        self.assertIn("Start six-score review", page.get_data(as_text=True))
        response = self.client.post(
            "/api/active-learning/sessions/" + sid + "/upgrade",
            json={"expected_round_id": old.round_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["data"]["round"]["review"]["categories"]), 6)
        self.assertEqual(store.get_round(old.round_id).analysis, old.analysis)


if __name__ == "__main__":
    unittest.main()
