import unittest

from app import create_app
from app.module_registry import MODULES, WORKFLOWS, list_modules, list_workflows


class FlaskShellRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_app_factory_creates_app(self):
        self.assertEqual(self.app.name, "app")

    def test_health_route(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["status"], "ok")

    def test_home_route(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/workflows/active-learning-dashboard/"
            )
        )

    def test_removed_mockup_route_returns_404(self):
        response = self.client.get("/mockups/final-dashboard/")

        self.assertEqual(response.status_code, 404)

    def test_modules_index_lists_planned_modules(self):
        response = self.client.get("/modules/")

        self.assertEqual(response.status_code, 200)
        for module in MODULES:
            self.assertIn(module.title.encode("utf-8"), response.data)

    def test_each_module_placeholder_page_loads(self):
        for module in MODULES:
            with self.subTest(module=module.slug):
                response = self.client.get(f"/modules/{module.slug}/")

                self.assertEqual(response.status_code, 200)
                self.assertIn(module.title.encode("utf-8"), response.data)

    def test_each_module_health_route_loads(self):
        for module in MODULES:
            with self.subTest(module=module.slug):
                response = self.client.get(f"/modules/{module.slug}/health")

                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json["ok"])
                self.assertEqual(response.json["data"]["module"], module.slug)

    def test_each_module_state_route_loads(self):
        for module in MODULES:
            with self.subTest(module=module.slug):
                response = self.client.get(f"/modules/{module.slug}/api/state")

                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json["ok"])
                self.assertEqual(response.json["data"]["module"], module.slug)

    def test_workflows_index_redirects_to_product_entry(self):
        response = self.client.get("/workflows/")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/workflows/active-learning-dashboard/"
            )
        )

    def test_active_learning_workflow_entry_loads(self):
        self.assertEqual(len(WORKFLOWS), 1)
        response = self.client.get("/workflows/active-learning-dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Active Learning Dashboard", response.data)

    def test_unknown_module_returns_404_envelope(self):
        response = self.client.get("/modules/not-real/")

        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "not_found")

    def test_enabled_module_filter(self):
        modules = list_modules(["selection", "rule-panel"])

        self.assertEqual([module.slug for module in modules], ["selection", "rule-panel"])

    def test_enabled_workflow_filter(self):
        workflows = list_workflows(["data-workspace", "projection"])

        self.assertEqual(workflows, ())

    def test_app_factory_mounts_only_enabled_module_pages(self):
        app = create_app(enabled_modules=["data-workspace"])
        client = app.test_client()

        data_workspace_response = client.get("/modules/data-workspace/")
        projection_response = client.get("/modules/projection/")

        self.assertEqual(data_workspace_response.status_code, 200)
        self.assertEqual(projection_response.status_code, 404)

    def test_app_factory_mounts_workflow_when_required_modules_are_enabled(self):
        app = create_app(enabled_modules=[module.slug for module in MODULES])
        client = app.test_client()

        response = client.get("/workflows/active-learning-dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Active Learning Dashboard", response.data)

    def test_app_factory_hides_workflow_when_required_modules_are_disabled(self):
        app = create_app(enabled_modules=["projection"])
        client = app.test_client()

        response = client.get("/workflows/active-learning-dashboard/")

        self.assertEqual(response.status_code, 404)

    def test_unknown_enabled_module_is_rejected(self):
        with self.assertRaises(ValueError):
            list_modules(["not-real"])

        with self.assertRaises(ValueError):
            create_app(enabled_modules=["not-real"])


if __name__ == "__main__":
    unittest.main()
