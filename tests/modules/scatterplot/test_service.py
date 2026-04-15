import unittest

from app.modules.algorithm_adapters.service import run_default_analysis
from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.labeling.schemas import LabelingState, ManualAnnotation
from app.modules.projection.service import project_feature_matrix
from app.modules.scatterplot.service import build_render_payload, selected_point_ids
from app.modules.selection.schemas import SelectionContext


class ScatterplotServiceTests(unittest.TestCase):
    def setUp(self):
        self.dataset = create_dataset(
            [
                {"point_id": "p1", "features": [0, 0], "metadata": {"label": "a"}},
                {"point_id": "p2", "features": [0.2, 0], "metadata": {"label": "a"}},
                {"point_id": "p3", "features": [5, 5], "metadata": {"label": "b"}},
                {"point_id": "p4", "features": [5.2, 5], "metadata": {"label": "b"}},
                {"point_id": "p5", "features": [10, 0], "metadata": {"label": "c"}},
                {"point_id": "p6", "features": [10.2, 0], "metadata": {"label": "c"}},
                {"point_id": "p_outlier", "features": [20, 20], "metadata": {"label": "outlier"}},
            ],
            dataset_id="scatter_test",
            feature_names=["x", "y"],
        )
        self.matrix = create_feature_matrix(self.dataset)
        self.projection = project_feature_matrix(self.matrix)
        self.analysis = run_default_analysis(
            self.matrix,
            n_clusters=2,
            outlier_n_neighbors=2,
            outlier_contamination=0.1,
        )

    def test_render_payload_includes_one_point_per_projection_coordinate(self):
        payload = build_render_payload(
            self.dataset,
            self.projection,
            self.analysis.cluster_result,
            self.analysis.outlier_result,
        )

        self.assertEqual(len(payload.points), len(self.dataset.points))
        self.assertEqual(
            [point.point_id for point in payload.points],
            [point.point_id for point in self.dataset.points],
        )
        self.assertIn("projection_id", payload.diagnostics)

    def test_render_payload_marks_selected_points(self):
        context = SelectionContext(
            dataset_id="scatter_test",
            selected_point_ids=("p1", "p3"),
            unselected_point_ids=("p2", "p4", "p5", "p6", "p_outlier"),
        )

        payload = build_render_payload(
            self.dataset,
            self.projection,
            self.analysis.cluster_result,
            self.analysis.outlier_result,
            selection_context=context,
        )

        self.assertEqual(selected_point_ids(payload), ("p1", "p3"))

    def test_render_payload_includes_outlier_and_manual_label_state(self):
        context = SelectionContext(
            dataset_id="scatter_test",
            selected_point_ids=("p_outlier",),
            unselected_point_ids=("p1", "p2", "p3", "p4", "p5", "p6"),
        )
        labeling_state = LabelingState(
            dataset_id="scatter_test",
            annotations=(
                ManualAnnotation(
                    annotation_id="annotation_001",
                    dataset_id="scatter_test",
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("p_outlier",),
                    label_type="outlier",
                    label_value=True,
                ),
            ),
        )

        payload = build_render_payload(
            self.dataset,
            self.projection,
            self.analysis.cluster_result,
            self.analysis.outlier_result,
            selection_context=context,
            labeling_state=labeling_state,
        )
        outlier_point = next(point for point in payload.points if point.point_id == "p_outlier")

        self.assertTrue(outlier_point.is_outlier)
        self.assertTrue(outlier_point.selected)
        self.assertEqual(outlier_point.manual_labels[0]["display_label"], "outlier")

    def test_mismatched_projection_is_rejected(self):
        other_dataset = create_dataset(
            [{"point_id": "other", "features": [0, 0]}],
            dataset_id="other",
            feature_names=["x", "y"],
        )
        other_projection = project_feature_matrix(create_feature_matrix(other_dataset))

        with self.assertRaises(ValueError):
            build_render_payload(
                self.dataset,
                other_projection,
                self.analysis.cluster_result,
                self.analysis.outlier_result,
            )


if __name__ == "__main__":
    unittest.main()
