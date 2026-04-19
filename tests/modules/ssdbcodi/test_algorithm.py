import unittest

from app.modules.ssdbcodi.algorithm import (
    assign_classes_by_weighted_distance,
    combined_outlier_score,
    compute_local_density_score,
    compute_similarity_score,
    core_distances,
    pairwise_euclidean,
    reachability_matrix,
    run_ssdbcodi_core,
    select_outliers_by_score,
)

import numpy as np


class SsdbcodiAlgorithmTests(unittest.TestCase):
    def test_pairwise_euclidean_is_symmetric_with_zero_diagonal(self):
        values = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])

        distances = pairwise_euclidean(values)

        self.assertAlmostEqual(distances[0, 0], 0.0)
        self.assertAlmostEqual(distances[0, 1], 5.0)
        self.assertAlmostEqual(distances[1, 0], 5.0)
        self.assertAlmostEqual(distances[0, 2], 1.0)

    def test_core_distance_returns_min_pts_neighbor_distance(self):
        values = np.array([[0.0], [1.0], [3.0], [10.0]])
        distances = pairwise_euclidean(values)

        c_dist = core_distances(distances, min_pts=2)

        self.assertAlmostEqual(c_dist[0], 3.0)
        self.assertAlmostEqual(c_dist[1], 2.0)

    def test_reachability_matrix_uses_max_of_core_and_distance(self):
        distances = np.array([[0.0, 1.0], [1.0, 0.0]])
        c_dist = np.array([0.5, 2.0])

        r_dist = reachability_matrix(distances, c_dist)

        self.assertAlmostEqual(r_dist[0, 0], 0.5)
        self.assertAlmostEqual(r_dist[0, 1], 2.0)
        self.assertAlmostEqual(r_dist[1, 0], 2.0)
        self.assertAlmostEqual(r_dist[1, 1], 2.0)

    def test_local_density_score_high_for_uniform_density(self):
        values = np.array([[0.0], [1.0], [2.0], [3.0]])
        distances = pairwise_euclidean(values)
        c_dist = core_distances(distances, min_pts=1)
        r_dist = reachability_matrix(distances, c_dist)

        l_score = compute_local_density_score(r_dist, min_pts=1)

        for value in l_score:
            self.assertGreater(value, 0.0)

    def test_similarity_score_uses_nearest_labeled_outlier(self):
        values = np.array([[0.0], [1.0], [10.0]])
        distances = pairwise_euclidean(values)

        sim = compute_similarity_score(distances, labeled_outlier_indices=(2,))

        self.assertAlmostEqual(sim[2], 1.0)
        self.assertGreater(sim[1], sim[0])

    def test_similarity_score_is_zero_without_labeled_outliers(self):
        values = np.array([[0.0], [1.0], [10.0]])
        distances = pairwise_euclidean(values)

        sim = compute_similarity_score(distances)

        self.assertEqual(tuple(sim), (0.0, 0.0, 0.0))

    def test_combined_outlier_score_uses_weights(self):
        r = np.array([1.0, 0.0])
        l = np.array([1.0, 0.0])
        sim = np.array([0.0, 1.0])

        t = combined_outlier_score(r, l, sim, alpha=0.4, beta=0.3)

        self.assertAlmostEqual(t[0], 0.0)
        self.assertAlmostEqual(t[1], 1.0)

    def test_select_outliers_picks_top_score(self):
        t_score = np.array([0.1, 0.9, 0.2, 0.5])

        outliers = select_outliers_by_score(t_score, contamination=0.25)

        self.assertEqual(outliers, (1,))

    def test_assign_classes_by_weighted_distance_picks_closest_class(self):
        values = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
                [10.2, 0.0],
            ]
        )
        distances = pairwise_euclidean(values)
        c_dist = core_distances(distances, min_pts=1)
        r_dist = reachability_matrix(distances, c_dist)

        labels, origin = assign_classes_by_weighted_distance(
            distances=distances,
            r_dist=r_dist,
            seeds={0: "A", 2: "B"},
            rscore_weight=0.5,
        )

        self.assertEqual(labels[0], "A")
        self.assertEqual(labels[1], "A")
        self.assertEqual(labels[2], "B")
        self.assertEqual(labels[3], "B")
        self.assertEqual(origin[0], 0)
        self.assertEqual(origin[2], 2)

    def test_assign_classes_respects_excluded_indices(self):
        values = np.array([[0.0], [1.0], [5.0]])
        distances = pairwise_euclidean(values)
        c_dist = core_distances(distances, min_pts=1)
        r_dist = reachability_matrix(distances, c_dist)

        labels, _ = assign_classes_by_weighted_distance(
            distances=distances,
            r_dist=r_dist,
            seeds={0: "A"},
            rscore_weight=0.5,
            excluded_indices=(2,),
        )

        self.assertEqual(labels[0], "A")
        self.assertEqual(labels[1], "A")
        self.assertIsNone(labels[2])

    def test_assign_classes_rejects_invalid_weight(self):
        values = np.array([[0.0], [1.0]])
        distances = pairwise_euclidean(values)
        c_dist = core_distances(distances, min_pts=1)
        r_dist = reachability_matrix(distances, c_dist)
        with self.assertRaises(ValueError):
            assign_classes_by_weighted_distance(
                distances=distances,
                r_dist=r_dist,
                seeds={0: "A"},
                rscore_weight=1.5,
            )

    def test_run_ssdbcodi_core_returns_expected_keys(self):
        values = [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [20.0, 20.0],
        ]

        result = run_ssdbcodi_core(
            values=values,
            seeds={0: "cluster_1", 3: "cluster_2"},
            min_pts=2,
            alpha=0.4,
            beta=0.3,
            contamination=0.15,
            rscore_weight=0.5,
        )

        self.assertEqual(set(result.keys()), {
            "assigned_label",
            "e_max", "r_score", "l_score", "sim_score", "t_score", "c_dist",
            "outlier_indices", "seed_origin",
            "labeled_outlier_indices", "min_pts", "alpha", "beta",
            "contamination", "rscore_weight",
        })
        self.assertEqual(len(result["assigned_label"]), len(values))
        self.assertGreaterEqual(len(result["outlier_indices"]), 1)
        self.assertIn(6, result["outlier_indices"])  # the [20, 20] point should be flagged
        self.assertEqual(result["assigned_label"][0], "cluster_1")
        self.assertEqual(result["assigned_label"][3], "cluster_2")
        for value in result["r_score"]:
            self.assertTrue(0.0 <= value <= 1.0)

    def test_run_ssdbcodi_core_rejects_invalid_min_pts(self):
        with self.assertRaises(ValueError):
            run_ssdbcodi_core(
                values=[[0.0], [1.0]],
                seeds={0: "A"},
                min_pts=0,
            )

    def test_run_ssdbcodi_core_rejects_no_seeds(self):
        with self.assertRaises(ValueError):
            run_ssdbcodi_core(values=[[0.0], [1.0]], seeds={}, min_pts=1)

    def test_run_ssdbcodi_core_rejects_invalid_rscore_weight(self):
        with self.assertRaises(ValueError):
            run_ssdbcodi_core(
                values=[[0.0], [1.0]],
                seeds={0: "A"},
                min_pts=1,
                rscore_weight=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
