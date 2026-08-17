import io
import json
import unittest

import numpy as np
from scipy.io import savemat

from app.modules.active_learning.data import (
    CsvDatasetAdapter,
    dataset_adapter,
    display_condition,
    import_csv_bytes,
    import_json_bytes,
    import_mat_bytes,
    prepare_records,
)


class ActiveLearningDataTests(unittest.TestCase):
    def test_mixed_records_are_stable_and_ground_truth_is_not_a_feature(self):
        records = [
            {"id": "r1", "age": 21, "color": "red", "score": None, "truth": "A"},
            {"id": "r2", "age": 40, "color": "blue", "score": 8.0, "truth": "B"},
            {"id": "r3", "age": 35, "color": None, "score": 4.0, "truth": "A"},
        ]
        first = prepare_records(
            records,
            dataset_id="mixed",
            point_id_column="id",
            feature_columns=("age", "color", "score"),
            ground_truth_columns=("truth",),
        )
        second = prepare_records(
            records,
            dataset_id="mixed",
            point_id_column="id",
            feature_columns=("age", "color", "score"),
            ground_truth_columns=("truth",),
        )

        self.assertEqual(
            first.version.dataset_version_id,
            second.version.dataset_version_id,
        )
        self.assertEqual(first.version.point_ids, ("r1", "r2", "r3"))
        self.assertNotIn("truth", first.feature_matrix.feature_names)
        self.assertTrue(
            any(name.startswith("encoded::color==") for name in first.feature_matrix.feature_names)
        )
        self.assertTrue(np.isfinite(np.asarray(first.feature_matrix.values)).all())
        self.assertEqual(first.raw_records[0]["ground_truth"], {"truth": "A"})

    def test_generated_point_ids_are_stable(self):
        records = [{"x": 1}, {"x": 2}]
        first = prepare_records(records)
        second = prepare_records(records)

        self.assertEqual(first.version.point_ids, second.version.point_ids)
        self.assertTrue(first.version.point_ids[0].startswith("p_"))

    def test_preprocessing_version_changes_do_not_change_generated_point_ids(self):
        records = [{"kind": None, "x": 1}, {"kind": "a", "x": 2}]
        first = prepare_records(records)
        second = prepare_records(
            records,
            preprocessing_config={"missing_category_token": "__not_recorded__"},
        )

        self.assertEqual(first.version.point_ids, second.version.point_ids)
        self.assertNotEqual(
            first.version.dataset_version_id,
            second.version.dataset_version_id,
        )
        self.assertEqual(
            first.version.content_fingerprint,
            second.version.content_fingerprint,
        )

    def test_dataset_adapter_registry_exposes_built_in_contracts(self):
        self.assertIsInstance(dataset_adapter("CSV"), CsvDatasetAdapter)
        with self.assertRaisesRegex(ValueError, "csv, json, or mat"):
            dataset_adapter("parquet")

    def test_duplicate_explicit_point_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_records(
                [{"id": "same", "x": 1}, {"id": "same", "x": 2}],
                point_id_column="id",
                feature_columns=("x",),
            )

    def test_csv_json_and_mat_use_the_same_prepared_contract(self):
        csv_dataset = import_csv_bytes(
            b"id,x,kind\np1,1,a\np2,2,b\n",
            point_id_column="id",
            feature_columns=("x", "kind"),
        )
        json_dataset = import_json_bytes(
            json.dumps(
                [{"id": "p1", "x": 1, "kind": "a"}, {"id": "p2", "x": 2, "kind": "b"}]
            ).encode("utf-8"),
            point_id_column="id",
            feature_columns=("x", "kind"),
        )
        buffer = io.BytesIO()
        savemat(buffer, {"X": np.asarray([[1.0, 2.0], [3.0, 4.0]]), "y": np.asarray([[1], [2]])})
        mat_dataset = import_mat_bytes(
            buffer.getvalue(),
            feature_names=("x", "y"),
        )

        for prepared in (csv_dataset, json_dataset, mat_dataset):
            self.assertEqual(prepared.version.point_count, 2)
            self.assertGreaterEqual(len(prepared.feature_matrix.feature_names), 2)

    def test_display_conditions_restore_raw_feature_meaning(self):
        prepared = prepare_records(
            [
                {"age": 10, "color": "red"},
                {"age": 20, "color": "blue"},
                {"age": 30, "color": "red"},
            ]
        )
        numeric = display_condition(
            "age",
            ">",
            0.0,
            prepared.version.transformation_map,
        )
        encoded_name = next(
            name for name in prepared.feature_matrix.feature_names if name.endswith("color==red")
        )
        categorical = display_condition(
            encoded_name,
            ">",
            0.5,
            prepared.version.transformation_map,
        )

        self.assertEqual(numeric["source_feature"], "age")
        self.assertNotIn("encoded::", numeric["display_text"])
        self.assertEqual(categorical["display_text"], "color = red")


if __name__ == "__main__":
    unittest.main()
