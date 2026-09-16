"""Real Wine workflow, isolated synthetic human feedback; no evaluation truth."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from werkzeug.serving import make_server, WSGIRequestHandler
from app import create_app
from app.modules.active_learning import ActiveLearningStore
from app.modules.ssdbcodi.service import run_ssdbcodi

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class QuietHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        pass


@unittest.skipIf(sync_playwright is None, "optional Playwright is not installed")
class WineWorkflowBrowserTests(unittest.TestCase):
    def test_wine_selection_feedback_corrections_revert_and_anchor_replacement(self):
        with tempfile.TemporaryDirectory(prefix="wine-regression-") as directory:
            app = create_app()
            app.config.update(TESTING=True, ACTIVE_LEARNING_DB_PATH=str(Path(directory) / "wine.sqlite"))
            client = app.test_client()
            location = client.post('/workflows/active-learning-dashboard/wine-fixture').headers['Location']
            sid = location.rstrip('/').split('/')[-1]
            state_url = f'/api/active-learning/sessions/{sid}/state'

            def state():
                return client.get(state_url).json['data']

            def pool(current, name):
                return next(c['pool_size'] for c in current['review']['categories'] if c['short'] == name)

            initial = state()
            self.assertEqual(len(initial['plot_points']), 129)
            self.assertEqual([pool(initial, key) for key in ('Coverage', 'Reach', 'Known')], [0, 0, 0])
            server = make_server('127.0.0.1', 0, app, threaded=True, request_handler=QuietHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True, executable_path=os.environ.get('METRIC_TEST_BROWSER_PATH') or None)
                    try:
                        page = browser.new_page(viewport={'width': 1542, 'height': 1000})
                        errors = []
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        page.on('dialog', lambda dialog: dialog.accept())
                        page.goto(f'http://127.0.0.1:{server.server_port}' + location)

                        def select(pid):
                            point = page.locator(f'.review-plot-point[data-point-id="{pid}"]')
                            point.focus()
                            point.press('Enter')

                        def label(action):
                            round_index = state()['round']['round_index']
                            page.locator(f'[data-label-action="{action}"]').click()
                            page.wait_for_function("expected => JSON.parse(document.getElementById('review-state').textContent).round.round_index === expected", arg=round_index + 1, timeout=60000)

                        select('wine_020')
                        for pid in ('wine_012', 'wine_064'):
                            page.get_by_role('checkbox', name=f'Include {pid} in label batch', exact=True).check()
                        self.assertEqual(page.locator('.review-plot-point.batch-selected').count(), 2)
                        self.assertIn('wine_020', page.locator('#label-scope-warning').inner_text())
                        page.locator('#point-search').fill('wine_0')
                        page.locator('#semantic-label').fill('AUDIT_TYPE_A')
                        label('semantic')
                        semantic = state()
                        self.assertEqual({e['point_id'] for e in semantic['active_labels']}, {'wine_012', 'wine_064'})
                        self.assertGreater(pool(semantic, 'Coverage'), 0)
                        self.assertGreater(pool(semantic, 'Reach'), 0)
                        self.assertEqual(pool(semantic, 'Known'), 0)
                        self.assertEqual(page.locator('#point-search').input_value(), 'wine_0')
                        page.locator('#status-filter').select_option('all')
                        self.assertIn('Reviewed', page.locator('#matrix-rows tr[data-point-id="wine_012"] .score-tile').nth(3).inner_text())

                        select('wine_074')
                        label('outlier')
                        self.assertGreater(pool(state(), 'Known'), 0)
                        label('normal')
                        corrected = state()
                        self.assertEqual(pool(corrected, 'Known'), 0)
                        self.assertFalse(next(p['is_outlier'] for p in corrected['plot_points'] if p['point_id'] == 'wine_074'))
                        self.assertIn('wine_074', corrected['analysis']['diagnostics']['normal_reference_point_ids'])
                        self.assertEqual(len([e for e in corrected['active_labels'] if e['point_id'] == 'wine_074']), 1)

                        select('wine_079')
                        label('uncertain')
                        self.assertEqual(next(r['review_status'] for r in state()['review']['rows'] if r['point_id'] == 'wine_079'), 'deferred')
                        self.assertNotIn('wine_079', state()['analysis']['diagnostics']['normal_reference_point_ids'])
                        select('wine_020')
                        label('normal')
                        self.assertEqual(next(r['review_status'] for r in state()['review']['rows'] if r['point_id'] == 'wine_079'), 'unlabeled')
                        current = state()
                        stale = client.post(f'/api/active-learning/sessions/{sid}/rounds/{initial["round"]["round_id"]}/labels', json={
                            'expected_round_id': initial['round']['round_id'], 'expected_label_revision': initial['round']['label_revision'],
                            'labels': [{'point_id': 'wine_030', 'label_dimension': 'outlier_status', 'label_value': True}]})
                        self.assertEqual(stale.status_code, 409)
                        self.assertEqual(state()['round']['round_id'], current['round']['round_id'])
                        page.locator(f'[data-revert="{initial["round"]["round_id"]}"]').click()
                        page.wait_for_function("JSON.parse(document.getElementById('review-state').textContent).round.round_index === 0", timeout=60000)
                        self.assertEqual(state()['review'], initial['review'])
                        self.assertEqual(state()['active_labels'], [])
                        self.assertEqual(page.locator('#point-search').input_value(), 'wine_0')
                        self.assertEqual(page.locator('#status-filter').input_value(), 'all')
                        self.assertEqual(errors, [])
                        if os.environ.get('METRIC_TEST_SCREENSHOT_DIR'):
                            page.screenshot(path=str(Path(os.environ['METRIC_TEST_SCREENSHOT_DIR']) / 'wine-review-fixed.png'), full_page=True)
                            page.locator('.review-table-scroll').screenshot(path=str(Path(os.environ['METRIC_TEST_SCREENSHOT_DIR']) / 'wine-matrix-fixed.png'))
                    finally:
                        browser.close()
            finally:
                server.shutdown()
                thread.join()

            store = ActiveLearningStore(app.config['ACTIVE_LEARNING_DB_PATH'])
            matrix = store.load_prepared_dataset(initial['session']['dataset_version_id']).feature_matrix
            anchors = [seed.point_id for seed in run_ssdbcodi(matrix).seeds]
            response = client.post(f'/api/active-learning/sessions/{sid}/rounds/{initial["round"]["round_id"]}/labels', json={
                'expected_round_id': initial['round']['round_id'], 'expected_label_revision': state()['round']['label_revision'],
                'labels': [{'point_id': pid, 'label_dimension': 'outlier_status', 'label_value': True} for pid in anchors]})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertTrue(set(anchors) <= {p['point_id'] for p in state()['plot_points'] if p['is_outlier']})
            self.assertFalse(set(anchors) & set(state()['analysis']['diagnostics']['normal_reference_point_ids']))
            self.assertEqual(len(state()['active_labels']), 3)
            json.dumps(state(), allow_nan=False)
