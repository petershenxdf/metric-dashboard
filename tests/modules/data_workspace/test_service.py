import unittest

from app.modules.data_workspace.fixtures import (
    iris_feature_names,
    iris_raw_points,
    tiny_feature_names,
    tiny_raw_points,
)
from app.modules.data_workspace.service import (
    create_dataset,
    create_feature_matrix,
    create_point_id_map,
)


class DataWorkspaceServiceTests(unittest.TestCase):
    def test_create_dataset_with_explicit_ids(self):
        dataset = create_dataset(
            iris_raw_points(),
            dataset_id="iris_sample",
            feature_names=iris_feature_names(),
        )

        self.assertEqual(dataset.dataset_id, "iris_sample")
        self.assertEqual(len(dataset.points), 15)
        self.assertEqual(dataset.points[0].point_id, "setosa_001")
        self.assertEqual(dataset.points[0].metadata["label"], "setosa")
        self.assertEqual(dataset.feature_names, tuple(iris_feature_names()))

    def test_create_dataset_generates_stable_ids(self):
        raw_points = [
            {"features": [1, 2]},
            {"features": [3, 4]},
            {"features": [5, 6]},
        ]

        dataset = create_dataset(raw_points)

        self.assertEqual([point.point_id for point in dataset.points], ["p1", "p2", "p3"])
        self.assertEqual(dataset.feature_names, ("feature_1", "feature_2"))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset([])

    def test_missing_features_are_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset([{"point_id": "p1"}])

    def test_inconsistent_feature_length_is_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset(
                [
                    {"point_id": "p1", "features": [1, 2]},
                    {"point_id": "p2", "features": [1, 2, 3]},
                ]
            )

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset(
                [
                    {"point_id": "p1", "features": [1, 2]},
                    {"point_id": "p1", "features": [3, 4]},
                ]
            )

    def test_non_numeric_features_are_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset([{"point_id": "p1", "features": [1, "bad"]}])

    def test_empty_feature_names_are_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset(
                [{"point_id": "p1", "features": [1, 2]}],
                feature_names=[],
            )

    def test_duplicate_feature_names_are_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset(
                [{"point_id": "p1", "features": [1, 2]}],
                feature_names=["x", "x"],
            )

    def test_empty_dataset_id_is_rejected(self):
        with self.assertRaises(ValueError):
            create_dataset(
                [{"point_id": "p1", "features": [1, 2]}],
                dataset_id="",
            )

    def test_feature_matrix_preserves_point_order(self):
        dataset = create_dataset(tiny_raw_points(), feature_names=tiny_feature_names())
        matrix = create_feature_matrix(dataset)

        self.assertEqual(matrix.point_ids, ("p1", "p2", "p3"))
        self.assertEqual(matrix.feature_names, ("x", "y"))
        self.assertEqual(matrix.values, ((0.0, 0.0), (1.0, 0.0), (4.0, 4.0)))

    def test_point_id_map_uses_stable_ids(self):
        dataset = create_dataset(tiny_raw_points(), feature_names=tiny_feature_names())
        point_map = create_point_id_map(dataset)

        self.assertEqual(set(point_map), {"p1", "p2", "p3"})
        self.assertEqual(point_map["p3"].metadata["label"], "b")

    def test_feature_matrix_rejects_non_dataset(self):
        with self.assertRaises(ValueError):
            create_feature_matrix("not a dataset")


if __name__ == "__main__":
    unittest.main()
