import unittest

from app.modules.data_workspace.service import create_dataset
from app.modules.selection.schemas import SelectionAction
from app.modules.selection.service import (
    apply_selection_action,
    clear_selection,
    create_selection_store,
    deselect_points,
    get_selection_context,
    list_selection_groups,
    replace_selection,
    save_selection_group,
    select_selection_group,
    select_points,
    delete_selection_group,
    toggle_points,
)


class SelectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.dataset = create_dataset(
            [
                {"point_id": "p1", "features": [0]},
                {"point_id": "p2", "features": [1]},
                {"point_id": "p3", "features": [2]},
            ],
            dataset_id="selection_test",
            feature_names=["x"],
        )
        self.store = create_selection_store(self.dataset)

    def test_select_points_adds_to_existing_selection(self):
        select_points(self.store, ["p1"])
        state = select_points(self.store, ["p2"])

        self.assertEqual(state.selected_point_ids, ("p1", "p2"))
        self.assertEqual(state.unselected_point_ids, ("p3",))

    def test_deselect_points_removes_points(self):
        replace_selection(self.store, ["p1", "p2"])
        state = deselect_points(self.store, ["p1"])

        self.assertEqual(state.selected_point_ids, ("p2",))

    def test_replace_selection_replaces_existing_points(self):
        select_points(self.store, ["p1", "p2"])
        state = replace_selection(self.store, ["p3"])

        self.assertEqual(state.selected_point_ids, ("p3",))

    def test_toggle_points_switches_membership(self):
        replace_selection(self.store, ["p1", "p2"])
        state = toggle_points(self.store, ["p2", "p3"])

        self.assertEqual(state.selected_point_ids, ("p1", "p3"))

    def test_clear_selection_allows_empty_selection(self):
        replace_selection(self.store, ["p1", "p2"])
        state = clear_selection(self.store)

        self.assertEqual(state.selected_point_ids, ())
        self.assertEqual(state.unselected_point_ids, ("p1", "p2", "p3"))

    def test_unknown_point_is_rejected(self):
        with self.assertRaises(ValueError):
            select_points(self.store, ["not-real"])

    def test_context_is_read_only_payload_for_downstream_modules(self):
        replace_selection(self.store, ["p1"])
        context = get_selection_context(self.store)

        self.assertEqual(context.source, "selection")
        self.assertEqual(context.selected_point_ids, ("p1",))
        self.assertEqual(context.unselected_point_ids, ("p2", "p3"))

    def test_apply_action_supports_future_source_and_mode_metadata(self):
        action = SelectionAction(
            action="replace",
            point_ids=("p1", "p2"),
            source="lasso",
            mode="replace",
            metadata={"gesture_id": "lasso_001"},
        )
        result = apply_selection_action(self.store, action)

        self.assertEqual(result.action.source, "lasso")
        self.assertEqual(result.action.metadata["gesture_id"], "lasso_001")
        self.assertEqual(result.state.selected_point_ids, ("p1", "p2"))

    def test_invalid_action_type_is_rejected(self):
        with self.assertRaises(ValueError):
            SelectionAction(action="paint", point_ids=("p1",))

    def test_save_selection_group_uses_current_selection_by_default(self):
        replace_selection(self.store, ["p1", "p3"])

        group = save_selection_group(self.store, "Interesting pair")

        self.assertEqual(group.group_id, "group_001")
        self.assertEqual(group.group_name, "Interesting pair")
        self.assertEqual(group.point_ids, ("p1", "p3"))
        self.assertEqual(list_selection_groups(self.store), (group,))

    def test_select_selection_group_replaces_current_selection(self):
        group = save_selection_group(self.store, "Right side", ["p2", "p3"])
        replace_selection(self.store, ["p1"])

        result = select_selection_group(self.store, group.group_id)

        self.assertEqual(result.action.source, "selection_group")
        self.assertEqual(result.action.metadata["group_name"], "Right side")
        self.assertEqual(result.state.selected_point_ids, ("p2", "p3"))

    def test_delete_selection_group_removes_saved_group(self):
        group = save_selection_group(self.store, "Temporary", ["p1"])

        deleted = delete_selection_group(self.store, group.group_id)

        self.assertEqual(deleted, group)
        self.assertEqual(list_selection_groups(self.store), ())

    def test_duplicate_selection_group_names_are_rejected_case_insensitively(self):
        save_selection_group(self.store, "Focus", ["p1"])

        with self.assertRaises(ValueError):
            save_selection_group(self.store, "focus", ["p2"])

    def test_empty_selection_group_is_rejected(self):
        with self.assertRaises(ValueError):
            save_selection_group(self.store, "Empty")


if __name__ == "__main__":
    unittest.main()
