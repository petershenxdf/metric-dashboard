import json
import unittest

from app.modules.labeling.schemas import LabelingState, ManualAnnotation
from app.modules.ssdbcodi.algorithm import run_ssdbcodi_core
from app.modules.ssdbcodi.service import bootstrap_seeds_from_kmeans, run_ssdbcodi
from app.shared.schemas import FeatureMatrix


class FeedbackRegressionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = FeatureMatrix(
            tuple(f"p{i}" for i in range(8)), ("x",),
            tuple((x,) for x in (0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1)),
        )

    def labels(self, ids, value):
        return LabelingState("test", (ManualAnnotation(
            "feedback", "test", "human", "selected_points", tuple(ids), "outlier", value),))

    def test_normal_support_changes_neighbor_scores_without_becoming_class_seed(self):
        options = dict(values=self.matrix.values, seeds={0: "A"}, min_pts=1)
        before = run_ssdbcodi_core(**options)
        after = run_ssdbcodi_core(**options, labeled_normal_indices=(6,))
        self.assertGreater(after["r_score"][7], before["r_score"][7])
        self.assertLess(after["t_score"][7], before["t_score"][7])
        self.assertNotIn(6, after["outlier_indices"])
        self.assertEqual(after["assigned_label"][6], "A")
        self.assertEqual(after["seed_origin"][6], 0)

    def test_normal_service_feedback_keeps_semantic_seeds_separate(self):
        before = run_ssdbcodi(self.matrix, n_clusters=1, min_pts=1)
        after = run_ssdbcodi(self.matrix, self.labels(["p6"], False), n_clusters=1, min_pts=1)
        self.assertEqual(before.seeds, after.seeds)
        self.assertNotIn("p6", [s.point_id for s in after.seeds])
        self.assertIn("p6", after.diagnostics["normal_reference_point_ids"])
        self.assertTrue(after.point_scores[6].is_reliable_normal)
        self.assertGreater(after.point_scores[7].r_score, before.point_scores[7].r_score)
        self.assertIsNotNone(after.point_scores[6].seed_origin_point_id)

    def test_core_never_flags_confirmed_normals_when_every_candidate_is_protected(self):
        result = run_ssdbcodi_core(self.matrix.values, {0: "A"}, min_pts=1,
                                   labeled_normal_indices=tuple(range(8)))
        self.assertEqual(result["outlier_indices"], ())
        self.assertEqual(set(result["assigned_label"]), {"A"})

    def test_core_rejects_conflicting_and_invalid_normal_indices(self):
        for normals in ((-1,), (8,), (True,)):
            with self.subTest(normals=normals), self.assertRaises(ValueError):
                run_ssdbcodi_core(self.matrix.values, {0: "A"}, min_pts=1, labeled_normal_indices=normals)
        with self.assertRaisesRegex(ValueError, "both confirmed"):
            run_ssdbcodi_core(self.matrix.values, {0: "A"}, min_pts=1,
                               labeled_normal_indices=(6,), labeled_outlier_indices=(6,))

    def test_reseeds_after_partial_and_complete_original_anchor_loss(self):
        original = bootstrap_seeds_from_kmeans(self.matrix, n_clusters=3, min_pts=1)
        for excluded in (tuple(original)[:1], tuple(original)):
            with self.subTest(excluded=excluded):
                result = run_ssdbcodi(self.matrix, self.labels([self.matrix.point_ids[i] for i in excluded], True), min_pts=1)
                self.assertEqual(len(result.seeds), 3)
                self.assertFalse({self.matrix.point_ids[i] for i in excluded} & {s.point_id for s in result.seeds})
                self.assertTrue({self.matrix.point_ids[i] for i in excluded} <= set(result.outlier_result.outlier_point_ids))
                again = run_ssdbcodi(self.matrix, self.labels([self.matrix.point_ids[i] for i in excluded], True), min_pts=1)
                self.assertEqual(result.to_dict(), again.to_dict())

    def test_one_remaining_candidate_reduces_bootstrap_size(self):
        result = run_ssdbcodi(self.matrix, self.labels(self.matrix.point_ids[:-1], True), min_pts=1)
        self.assertEqual([s.point_id for s in result.seeds], ["p7"])
        self.assertFalse(result.point_scores[-1].is_outlier)

    def test_all_confirmed_outliers_have_no_fake_normal_reference(self):
        result = run_ssdbcodi(self.matrix, self.labels(self.matrix.point_ids, True), min_pts=1)
        self.assertEqual(result.seeds, ())
        self.assertEqual(result.cluster_result.assignments, ())
        self.assertTrue(all(p.is_outlier and p.r_score == 0 and p.e_max is None for p in result.point_scores))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_outlier_correction_removes_normal_support(self):
        result = run_ssdbcodi(self.matrix, self.labels(["p6"], True), n_clusters=1, min_pts=1)
        self.assertNotIn("p6", result.diagnostics["normal_reference_point_ids"])
        self.assertTrue(result.point_scores[6].is_outlier)
        self.assertFalse(result.point_scores[6].is_reliable_normal)
