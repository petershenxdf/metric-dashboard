"""Optional real-browser checks: python -m unittest discover -s tests/browser -v.

Install playwright and its Chromium browser first (not a runtime dependency).
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from werkzeug.serving import make_server, WSGIRequestHandler

from app import create_app
from app.modules.active_learning import ActiveLearningService, ActiveLearningStore

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class QuietHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        pass


@unittest.skipIf(sync_playwright is None, "optional Playwright is not installed")
class ReviewMatrixBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            headless=True, executable_path=os.environ.get("METRIC_TEST_BROWSER_PATH") or None
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = Path(self.tempdir.name) / "test.sqlite3"
        self.service = ActiveLearningService(ActiveLearningStore(path))
        prepared = self.service.import_records(
            [
                {"id": f"p{i:02d}", "x": i // 8 * 6 + i % 4, "y": i % 3}
                for i in range(24)
            ],
            point_id_column="id",
        )
        session = self.service.create_session(prepared.version.dataset_version_id)
        self.sid = session.session_id
        app = create_app()
        app.config["ACTIVE_LEARNING_DB_PATH"] = str(path)
        self.server = make_server(
            "127.0.0.1", 0, app, threaded=True, request_handler=QuietHandler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.page = self.browser.new_page(viewport={"width": 1440, "height": 1100})
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(
            f"http://127.0.0.1:{self.server.server_port}/workflows/active-learning-dashboard/{self.sid}/"
        )
        self.page.wait_for_selector("#matrix-rows tr")

    def tearDown(self):
        self.page.close()
        self.server.shutdown()
        self.thread.join()
        self.tempdir.cleanup()
        self.assertEqual(self.errors, [])

    def test_pure_view_sort_filters_ties_na_and_exact_color_boundaries(self):
        actual = self.page.evaluate("""() => {
          const row = (id, a, b, duplicate=id) => ({point_id:id, group:'A', review_status:'unlabeled', duplicate_key:duplicate,
            scores:{a:{percentile:a,recommendable:a!==null},b:{percentile:b,recommendable:b!==null}}});
          const rows=[row('b',90,20,1),row('a',90,80,1),row('c',null,95),row('d',74.9999,null)];
          const before=JSON.stringify(rows);
          const filters={status:'unlabeled',group:'',search:'',recommendable:false,category:'a',mode:'all',
            conditions:[{category:'a',kind:'minimum',value:75},{category:'b',kind:'minimum',value:75}]};
          return {desc:ReviewMatrix.sortRows(rows,'a').map(r=>r.point_id),asc:ReviewMatrix.sortRows(rows,'a',false).map(r=>r.point_id),
            all:ReviewMatrix.filterRows(rows,filters).map(r=>r.point_id),
            any:ReviewMatrix.filterRows(rows,{...filters,mode:'any'}).map(r=>r.point_id),
            na:ReviewMatrix.filterRows(rows,{...filters,conditions:[{category:'a',kind:'na'}]}).map(r=>r.point_id),
            batch:ReviewMatrix.batch(rows,'a',4),unchanged:before===JSON.stringify(rows),
            bands:[null,24.9999,25,49.9999,50,74.9999,75,89.9999,90,100].map(ReviewMatrix.band)};
        }""")
        self.assertEqual(actual["desc"], ["a", "b", "d", "c"])
        self.assertEqual(actual["asc"], ["d", "a", "b", "c"])
        self.assertEqual(actual["all"], ["a"])
        self.assertEqual(actual["any"], ["b", "a", "c"])
        self.assertEqual(actual["na"], ["c"])
        self.assertEqual(actual["batch"], ["a", "d"])
        self.assertTrue(actual["unchanged"])
        self.assertEqual(
            actual["bands"],
            [
                "band-na",
                "band-0",
                "band-1",
                "band-1",
                "band-2",
                "band-2",
                "band-3",
                "band-3",
                "band-4",
                "band-4",
            ],
        )

    def test_linked_keyboard_selection_hidden_selection_and_filter_reset(self):
        point = self.page.locator('.review-plot-point[data-point-id="p00"]')
        point.focus()
        self.page.keyboard.press("Enter")
        self.assertIn("p00", self.page.locator("#selected-id").inner_text())
        self.assertIn("selected", point.get_attribute("class"))
        self.page.locator("#point-search").fill("missing-id")
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 0)
        self.assertIn("hidden", self.page.locator("#selected-hidden").inner_text())
        self.assertTrue(self.page.locator("#matrix-empty").is_visible())
        self.page.locator("#reset-filters").click()
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 24)
        self.page.locator('#matrix-rows tr[data-point-id="p01"] .score-tile').nth(
            3
        ).click()
        self.assertEqual(
            self.page.locator("#point-details h3").inner_text(), "Local Sparsity"
        )
        self.assertIn(
            "selected",
            self.page.locator('.review-plot-point[data-point-id="p01"]').get_attribute(
                "class"
            ),
        )

    def test_filters_keep_tiles_frozen_and_na_conditions_work(self):
        snapshot = self.page.locator("#review-state").text_content()
        self.page.locator("#condition-category").select_option("label_coverage_gap")
        self.page.locator("#condition-kind").select_option("na")
        self.page.locator("#add-condition").click()
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 24)
        self.page.locator("#condition-category").select_option("local_sparsity")
        self.page.locator("#condition-kind").select_option("minimum")
        self.page.locator("#condition-value").fill("100")
        self.page.locator("#add-condition").click()
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 0)
        self.page.locator("#condition-mode").select_option("any")
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 24)
        self.assertEqual(self.page.locator("#review-state").text_content(), snapshot)

    def test_label_submission_refreshes_round_and_retains_selected_point(self):
        self.page.locator('.review-plot-point[data-point-id="p00"]').focus()
        self.page.keyboard.press("Enter")
        with self.page.expect_navigation():
            self.page.locator('[data-label-action="outlier"]').click()
        self.page.wait_for_selector("#matrix-rows tr")
        state = json.loads(self.page.locator("#review-state").text_content())
        self.assertEqual(state["round"]["round_index"], 1)
        self.assertIn("p00", self.page.locator("#selected-id").inner_text())
        self.assertIn("hidden", self.page.locator("#selected-hidden").inner_text())
        self.assertTrue(state["active_labels"][0]["label_value"])
        self.assertGreater(
            next(
                c["pool_size"]
                for c in state["review"]["categories"]
                if c["id"] == "known_outlier_similarity"
            ),
            0,
        )

    def test_request_failure_keeps_selection_and_shows_error(self):
        self.page.route(
            "**/labels",
            lambda route: route.fulfill(
                status=409,
                content_type="application/json",
                body=json.dumps({"ok": False, "error": {"message": "Round changed"}}),
            ),
        )
        self.page.locator('.review-plot-point[data-point-id="p00"]').focus()
        self.page.keyboard.press("Enter")
        self.page.locator('[data-label-action="normal"]').click()
        self.page.wait_for_function(
            "document.getElementById('action-message').textContent.includes('Round changed')"
        )
        self.assertIn("p00", self.page.locator("#selected-id").inner_text())
        self.assertTrue(self.page.locator('[data-label-action="normal"]').is_enabled())

    def test_mobile_matrix_scroll_and_screenshot(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.assertEqual(
            self.page.locator(".review-matrix th[data-category-heading]").count(), 6
        )
        self.assertEqual(
            self.page.locator(".review-table-scroll").evaluate(
                "el => getComputedStyle(el).overflowX"
            ),
            "auto",
        )
        self.page.set_viewport_size({"width": 1440, "height": 1100})
        self.page.locator("#sort-category").select_option("local_sparsity")
        self.page.locator("#matrix-rows .score-tile").nth(3).click()
        self.page.screenshot(path=str(Path(self.tempdir.name) / "review.png"), full_page=True)

    def test_batch_mode_highlights_targets_and_requires_explicit_confirmation(self):
        self.page.locator('.review-plot-point[data-point-id="p02"]').focus()
        self.page.keyboard.press("Enter")
        for pid in ("p00", "p01"):
            self.page.get_by_role("checkbox", name=f"Include {pid} in label batch", exact=True).check()
        self.assertEqual(self.page.locator("#label-scope").input_value(), "batch")
        self.assertEqual(self.page.locator(".review-plot-point.batch-selected").count(), 2)
        self.assertIn("p02", self.page.locator("#label-scope-warning").inner_text())
        initial = self.page.locator("#review-state").text_content()
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator('[data-label-action="normal"]').click()
        self.assertEqual(self.page.locator("#review-state").text_content(), initial)
        self.page.once("dialog", lambda dialog: dialog.accept())
        with self.page.expect_navigation():
            self.page.locator('[data-label-action="normal"]').click()
        state = json.loads(self.page.locator("#review-state").text_content())
        self.assertEqual({e["point_id"] for e in state["active_labels"]}, {"p00", "p01"})
        self.assertEqual(self.page.locator("#label-scope").input_value(), "single")
        self.assertEqual(self.page.locator(".review-plot-point.batch-selected").count(), 0)

    def test_empty_batch_does_not_fall_back_to_focused_point(self):
        self.page.locator('.review-plot-point[data-point-id="p02"]').focus()
        self.page.keyboard.press("Enter")
        self.page.locator("#label-scope").select_option("batch")
        self.assertTrue(self.page.locator('[data-label-action="normal"]').is_disabled())
        self.page.get_by_role("checkbox", name="Include p00 in label batch", exact=True).check()
        self.page.locator("#use-focused-point").click()
        self.assertEqual(self.page.locator(".review-plot-point.batch-selected").count(), 0)
        self.assertFalse(self.page.get_by_role("checkbox", name="Include p00 in label batch", exact=True).is_checked())
        self.assertIn("SINGLE", self.page.locator("#label-targets").inner_text())
        self.assertIn("p02", self.page.locator("#label-targets").inner_text())

    def test_complete_filter_state_survives_commit_and_reload(self):
        initial = json.loads(self.page.locator("#review-state").text_content())
        row = next(r for r in initial["review"]["rows"] if r["group"] != "outlier")
        self.page.locator(f'.review-plot-point[data-point-id="{row["point_id"]}"]').focus()
        self.page.keyboard.press("Enter")
        self.page.locator("#status-filter").select_option("all")
        self.page.locator("#group-filter").select_option(row["group"])
        self.page.locator("#point-search").fill("p0")
        self.page.locator("#sort-category").select_option("local_sparsity")
        self.page.locator("#sort-direction").click()
        self.page.locator("#recommendable-only").check()
        self.page.locator("#condition-category").select_option("label_coverage_gap")
        self.page.locator("#condition-kind").select_option("na")
        self.page.locator("#add-condition").click()
        self.page.locator("#condition-category").select_option("model_instability")
        self.page.locator("#condition-kind").select_option("minimum")
        self.page.locator("#condition-value").fill("75.5")
        self.page.locator("#add-condition").click()
        self.page.locator("#condition-mode").select_option("any")
        before = self.page.evaluate("JSON.parse(new URL(location.href).searchParams.get('view'))")
        with self.page.expect_navigation():
            self.page.locator('[data-label-action="normal"]').click()
        for _ in range(2):
            after = self.page.evaluate("JSON.parse(new URL(location.href).searchParams.get('view'))")
            self.assertEqual(after, before)
            self.assertEqual(self.page.locator("#status-filter").input_value(), "all")
            self.assertEqual(self.page.locator("#group-filter").input_value(), row["group"])
            self.assertEqual(self.page.locator("#point-search").input_value(), "p0")
            self.assertIn("Lowest", self.page.locator("#sort-direction").inner_text())
            self.page.reload()

    def test_na_reasons_reviewed_raw_score_and_zero_instability_explanation(self):
        self.assertIn("human references", self.page.locator("#review-guidance").inner_text())
        self.assertIn("Needs semantic label", self.page.locator('#matrix-rows tr').first.locator('.score-tile').nth(1).inner_text())
        self.page.locator('.review-plot-point[data-point-id="p00"]').focus()
        self.page.keyboard.press("Enter")
        with self.page.expect_navigation():
            self.page.locator('[data-label-action="normal"]').click()
        self.page.locator("#status-filter").select_option("all")
        tile = self.page.locator('#matrix-rows tr[data-point-id="p00"] .score-tile').nth(3)
        self.assertIn("Reviewed", tile.inner_text())
        tile.click()
        self.assertIn("Raw score:", self.page.locator("#point-details").inner_text())
        self.assertIn("Outside the ranking pool", self.page.locator("#point-details").inner_text())
        explanation = self.page.evaluate("ReviewMatrix.scoreExplanation('model_instability', {raw:0,evidence:{}})")
        self.assertIn("agree", explanation)
        self.assertNotIn("disagree", explanation)

    def test_invalid_saved_view_is_ignored_safely(self):
        self.page.goto(self.page.url.split('?')[0] + '?view=%7Bbroken')
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 24)

    def test_hidden_batch_targets_survive_failed_request_and_can_be_cleared(self):
        self.page.get_by_role("checkbox", name="Include p00 in label batch", exact=True).check()
        self.page.locator("#point-search").fill("p01")
        self.assertIn("hidden by filters", self.page.locator("#label-targets").inner_text())
        self.page.route("**/labels", lambda route: route.fulfill(status=409, content_type="application/json",
                        body=json.dumps({"ok": False, "error": {"message": "Round changed"}})))
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator('[data-label-action="normal"]').click()
        self.page.wait_for_function("document.getElementById('action-message').textContent.includes('Round changed')")
        self.assertEqual(self.page.locator("#label-scope").input_value(), "batch")
        self.assertEqual(self.page.locator(".review-plot-point.batch-selected").count(), 1)
        self.page.locator("#point-search").fill("")
        self.assertTrue(self.page.get_by_role("checkbox", name="Include p00 in label batch", exact=True).is_checked())
        self.page.locator("#clear-batch").click()
        self.assertEqual(self.page.locator(".review-plot-point.batch-selected").count(), 0)

    def test_missing_saved_group_is_not_silently_broadened(self):
        from urllib.parse import quote
        view = quote(json.dumps({"group": "group_no_longer_present", "status": "all"}))
        self.page.goto(self.page.url.split('?')[0] + '?view=' + view)
        self.assertEqual(self.page.locator("#group-filter").input_value(), "group_no_longer_present")
        self.assertEqual(self.page.locator("#matrix-rows tr").count(), 0)
        self.assertTrue(self.page.locator("#matrix-empty").is_visible())


if __name__ == "__main__":
    unittest.main()
