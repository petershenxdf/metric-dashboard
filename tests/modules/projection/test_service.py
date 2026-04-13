import math
import unittest

from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.projection.service import project_feature_matrix, scaled_projection_points
from app.shared.schemas import FeatureMatrix


class ProjectionServiceTests(unittest.TestCase):
    def test_projection_creates_one_coordinate_per_point(self):
        feature_matrix = _square_feature_matrix()

        projection = project_feature_matrix(feature_matrix)

        self.assertEqual(projection.method, "mds")
        self.assertEqual(
            [coordinate.point_id for coordinate in projection.coordinates],
            ["p1", "p2", "p3", "p4"],
        )

    def test_projection_coordinates_are_finite(self):
        projection = project_feature_matrix(_square_feature_matrix())

        for coordinate in projection.coordinates:
            self.assertTrue(math.isfinite(coordinate.x))
            self.assertTrue(math.isfinite(coordinate.y))

    def test_projection_is_stable_for_same_input(self):
        first = project_feature_matrix(_square_feature_matrix())
        second = project_feature_matrix(_square_feature_matrix())

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_projection_preserves_distance_shape_for_square(self):
        projection = project_feature_matrix(_square_feature_matrix())
        coordinates = {coordinate.point_id: coordinate for coordinate in projection.coordinates}

        adjacent = _distance(coordinates["p1"], coordinates["p2"])
        diagonal = _distance(coordinates["p1"], coordinates["p4"])

        self.assertGreater(diagonal, adjacent)

    def test_invalid_input_type_is_rejected(self):
        with self.assertRaises(ValueError):
            project_feature_matrix("not a feature matrix")

    def test_invalid_matrix_values_are_rejected_by_schema(self):
        with self.assertRaises(ValueError):
            FeatureMatrix(
                point_ids=("p1",),
                feature_names=("x",),
                values=((float("nan"),),),
            )

    def test_scaled_projection_points_include_screen_coordinates_and_labels(self):
        projection = project_feature_matrix(_square_feature_matrix())

        points = scaled_projection_points(
            projection,
            {"p1": "corner", "p2": "corner", "p3": "corner", "p4": "corner"},
        )

        self.assertEqual(len(points), 4)
        self.assertEqual(points[0]["label"], "corner")
        self.assertIn("screen_x", points[0])
        self.assertIn("screen_y", points[0])
        self.assertTrue(0 <= points[0]["screen_x"] <= 860)
        self.assertTrue(0 <= points[0]["screen_y"] <= 520)


def _square_feature_matrix():
    dataset = create_dataset(
        [
            {"point_id": "p1", "features": [0, 0]},
            {"point_id": "p2", "features": [1, 0]},
            {"point_id": "p3", "features": [0, 1]},
            {"point_id": "p4", "features": [1, 1]},
        ],
        dataset_id="square",
        feature_names=["x", "y"],
    )
    return create_feature_matrix(dataset)


def _distance(first, second):
    return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)


if __name__ == "__main__":
    unittest.main()
