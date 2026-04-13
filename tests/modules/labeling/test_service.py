import unittest

from app.modules.labeling.service import (
    annotation_to_instruction,
    apply_labeling_action,
    clear_annotations,
    create_labeling_store,
    get_labeling_state,
)
from app.modules.selection.schemas import SelectionContext


class LabelingServiceTests(unittest.TestCase):
    def setUp(self):
        self.context = SelectionContext(
            dataset_id="labeling_test",
            selected_point_ids=("p1", "p2"),
            unselected_point_ids=("p3",),
        )
        self.store = create_labeling_store(self.context.dataset_id)

    def test_selected_points_become_assign_cluster_instruction(self):
        annotation = apply_labeling_action(
            self.store,
            self.context,
            action="assign_cluster",
            label_value="cluster_2",
        )

        instruction = annotation_to_instruction(annotation)
        self.assertEqual(annotation.point_ids, ("p1", "p2"))
        self.assertEqual(annotation.label_type, "cluster")
        self.assertEqual(instruction.instruction_type, "assign_cluster")
        self.assertEqual(instruction.parameters["target_label"], "cluster_2")

    def test_selected_points_become_outlier_instruction(self):
        annotation = apply_labeling_action(self.store, self.context, action="mark_outlier")

        instruction = annotation_to_instruction(annotation)
        self.assertEqual(annotation.label_type, "outlier")
        self.assertTrue(annotation.label_value)
        self.assertEqual(instruction.instruction_type, "is_outlier")

    def test_selected_points_become_not_outlier_instruction(self):
        annotation = apply_labeling_action(self.store, self.context, action="mark_not_outlier")

        instruction = annotation_to_instruction(annotation)
        self.assertFalse(annotation.label_value)
        self.assertEqual(instruction.instruction_type, "not_outlier")

    def test_empty_selection_is_rejected(self):
        empty_context = SelectionContext(
            dataset_id="labeling_test",
            selected_point_ids=(),
            unselected_point_ids=("p1",),
        )

        with self.assertRaises(ValueError):
            apply_labeling_action(self.store, empty_context, action="mark_outlier")

    def test_unknown_point_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_labeling_action(
                self.store,
                self.context,
                action="assign_cluster",
                label_value="cluster_2",
                point_ids=["not-real"],
            )

    def test_unselected_point_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_labeling_action(
                self.store,
                self.context,
                action="assign_cluster",
                label_value="cluster_2",
                point_ids=["p3"],
            )

    def test_annotation_history_and_reset(self):
        apply_labeling_action(self.store, self.context, action="assign_new_class", label_value="interesting")
        state = get_labeling_state(self.store)

        self.assertEqual(state.dataset_id, "labeling_test")
        self.assertEqual(len(state.annotations), 1)

        cleared = clear_annotations(self.store)
        self.assertEqual(len(cleared.annotations), 0)


if __name__ == "__main__":
    unittest.main()
