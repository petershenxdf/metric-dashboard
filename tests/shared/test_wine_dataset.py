import unittest

from app.shared.wine_dataset import (
    WINE_DATASET_ID,
    WINE_FEATURE_NAMES,
    load_wine_dataset,
    load_wine_feature_matrix,
    wine_mat_path,
)


class WineDatasetTests(unittest.TestCase):
    def test_loads_uploaded_wine_mat_with_raw_feature_names(self):
        dataset = load_wine_dataset()
        matrix = load_wine_feature_matrix()

        self.assertTrue(wine_mat_path().exists())
        self.assertEqual(dataset.dataset_id, WINE_DATASET_ID)
        self.assertEqual(len(dataset.points), 129)
        self.assertEqual(dataset.feature_names, WINE_FEATURE_NAMES)
        self.assertEqual(matrix.feature_names, WINE_FEATURE_NAMES)
        self.assertEqual(len(matrix.point_ids), 129)
        self.assertEqual(matrix.point_ids[0], "wine_001")
        self.assertEqual(dataset.points[0].metadata["source_file"], "wine.mat")
        self.assertIn("class_label", dataset.points[0].metadata)


if __name__ == "__main__":
    unittest.main()
