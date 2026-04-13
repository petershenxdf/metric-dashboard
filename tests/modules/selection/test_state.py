import unittest

from app.modules.data_workspace.service import create_dataset
from app.modules.selection.state import get_debug_store_for_dataset, reset_debug_store_for_dataset


class SelectionStateTests(unittest.TestCase):
    def test_dataset_store_rebuilds_when_point_ids_change_for_same_dataset_id(self):
        first_dataset = create_dataset(
            [{"point_id": "p1", "features": [0]}],
            dataset_id="same_id",
            feature_names=["x"],
        )
        second_dataset = create_dataset(
            [{"point_id": "p2", "features": [1]}],
            dataset_id="same_id",
            feature_names=["x"],
        )

        reset_debug_store_for_dataset(first_dataset, ["p1"])
        store = get_debug_store_for_dataset(second_dataset, ["p2"])

        self.assertEqual(store.known_point_ids, ("p2",))
        self.assertEqual(store.ordered_selected_point_ids(), ("p2",))


if __name__ == "__main__":
    unittest.main()
