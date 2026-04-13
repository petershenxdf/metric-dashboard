import unittest

from app import create_app
from app.module_registry import MODULES, WORKFLOWS, list_modules


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

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Metric Dashboard", response.data)

    def test_final_dashboard_mockup_route(self):
        response = self.client.get("/mockups/final-dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Final Dashboard Mockup", response.data)
        self.assertIn(b"Scatterplot", response.data)
        self.assertIn(b"Structured Instruction", response.data)

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

    def test_workflows_index_lists_workflows(self):
        response = self.client.get("/workflows/")

        self.assertEqual(response.status_code, 200)
        for workflow in WORKFLOWS:
            self.assertIn(workflow.title.encode("utf-8"), response.data)

    def test_each_workflow_placeholder_page_loads(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.slug):
                response = self.client.get(f"/workflows/{workflow.slug}/")

                self.assertEqual(response.status_code, 200)
                self.assertIn(workflow.title.encode("utf-8"), response.data)

    def test_unknown_module_returns_404_envelope(self):
        response = self.client.get("/modules/not-real/")

        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "not_found")

    def test_enabled_module_filter(self):
        modules = list_modules(["chatbox", "intent-instruction"])

        self.assertEqual([module.slug for module in modules], ["chatbox", "intent-instruction"])

    def test_unknown_enabled_module_is_rejected(self):
        with self.assertRaises(ValueError):
            list_modules(["not-real"])


if __name__ == "__main__":
    unittest.main()
