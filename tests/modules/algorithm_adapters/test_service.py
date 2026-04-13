import unittest

from app.modules.algorithm_adapters.service import (
    cluster_non_outliers,
    detect_outliers,
    run_default_analysis,
)
from app.modules.data_workspace.service import create_dataset, create_feature_matrix


class AlgorithmAdapterServiceTests(unittest.TestCase):
    def test_outlier_detection_preserves_point_ids(self):
        matrix = _outlier_feature_matrix()

        result = detect_outliers(matrix, n_neighbors=2, contamination=0.1)

        self.assertEqual([score.point_id for score in result.scores], list(matrix.point_ids))
        self.assertIn("p_outlier", result.outlier_point_ids)

    def test_clustering_excludes_detected_outliers(self):
        matrix = _outlier_feature_matrix()

        analysis = run_default_analysis(
            matrix,
            n_clusters=2,
            outlier_n_neighbors=2,
            outlier_contamination=0.1,
        )
        assigned_point_ids = {
            assignment.point_id
            for assignment in analysis.cluster_result.assignments
        }

        self.assertIn("p_outlier", analysis.outlier_result.outlier_point_ids)
        self.assertNotIn("p_outlier", assigned_point_ids)
        self.assertEqual(analysis.diagnostics["execution_order"], ["local_outlier_factor", "kmeans_on_non_outliers"])

    def test_cluster_count_is_adjustable(self):
        matrix = _cluster_feature_matrix()

        two_clusters = run_default_analysis(
            matrix,
            n_clusters=2,
            outlier_n_neighbors=2,
            outlier_contamination=0.1,
        )
        three_clusters = run_default_analysis(
            matrix,
            n_clusters=3,
            outlier_n_neighbors=2,
            outlier_contamination=0.1,
        )

        self.assertEqual(two_clusters.cluster_result.n_clusters, 2)
        self.assertEqual(three_clusters.cluster_result.n_clusters, 3)

    def test_unknown_outlier_ids_are_rejected(self):
        matrix = _cluster_feature_matrix()

        with self.assertRaises(ValueError):
            cluster_non_outliers(matrix, ["not-real"], n_clusters=2)

    def test_too_many_clusters_are_rejected(self):
        matrix = _cluster_feature_matrix()

        with self.assertRaises(ValueError):
            cluster_non_outliers(matrix, [], n_clusters=99)

    def test_non_integer_cluster_count_is_rejected(self):
        matrix = _cluster_feature_matrix()

        with self.assertRaises(ValueError):
            cluster_non_outliers(matrix, [], n_clusters="2")


def _outlier_feature_matrix():
    dataset = create_dataset(
        [
            {"point_id": "p1", "features": [0, 0]},
            {"point_id": "p2", "features": [0.1, 0]},
            {"point_id": "p3", "features": [0, 0.1]},
            {"point_id": "p4", "features": [5, 5]},
            {"point_id": "p5", "features": [5.1, 5]},
            {"point_id": "p6", "features": [5, 5.1]},
            {"point_id": "p_outlier", "features": [20, 20]},
        ],
        dataset_id="outlier_fixture",
        feature_names=["x", "y"],
    )
    return create_feature_matrix(dataset)


def _cluster_feature_matrix():
    dataset = create_dataset(
        [
            {"point_id": "p1", "features": [0, 0]},
            {"point_id": "p2", "features": [0.1, 0]},
            {"point_id": "p3", "features": [5, 5]},
            {"point_id": "p4", "features": [5.1, 5]},
            {"point_id": "p5", "features": [10, 0]},
            {"point_id": "p6", "features": [10.1, 0]},
        ],
        dataset_id="cluster_fixture",
        feature_names=["x", "y"],
    )
    return create_feature_matrix(dataset)


if __name__ == "__main__":
    unittest.main()
