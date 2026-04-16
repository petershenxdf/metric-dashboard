import unittest

from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.labeling.schemas import ManualAnnotation, LabelingState
from app.modules.ssdbcodi.fixtures import (
    CIRCLES_FIXTURE_DATASET_ID,
    ssdbcodi_dataset_id,
    ssdbcodi_feature_names,
    ssdbcodi_raw_points,
)
from app.modules.ssdbcodi.schemas import PointScores, SeedRecord, SsdbcodiResult
from app.modules.ssdbcodi.service import (
    bootstrap_seeds_from_kmeans,
    cluster_counts,
    collect_seeds_from_labeling,
    run_ssdbcodi,
)
from app.modules.ssdbcodi.store import SsdbcodiStore


class SsdbcodiServiceTests(unittest.TestCase):
    def setUp(self):
        dataset = create_dataset(
            ssdbcodi_raw_points(),
            dataset_id=ssdbcodi_dataset_id(),
            feature_names=ssdbcodi_feature_names(),
        )
        self.matrix = create_feature_matrix(dataset)

    def test_bootstrap_seeds_from_kmeans_returns_one_seed_per_cluster(self):
        seeds = bootstrap_seeds_from_kmeans(self.matrix, n_clusters=3)

        self.assertEqual(len(seeds), 3)
        labels = sorted(set(seeds.values()))
        self.assertEqual(labels, ["cluster_1", "cluster_2", "cluster_3"])
        for index in seeds:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, len(self.matrix.point_ids))
            self.assertFalse(self.matrix.point_ids[index].startswith("outlier_"))

    def test_circle_bootstrap_does_not_promote_fixture_outliers(self):
        dataset = create_dataset(
            ssdbcodi_raw_points(CIRCLES_FIXTURE_DATASET_ID),
            dataset_id=CIRCLES_FIXTURE_DATASET_ID,
            feature_names=ssdbcodi_feature_names(),
        )
        matrix = create_feature_matrix(dataset)

        seeds = bootstrap_seeds_from_kmeans(matrix, n_clusters=3)

        seed_point_ids = {matrix.point_ids[index] for index in seeds}
        self.assertFalse(any(point_id.startswith("circle_outlier_") for point_id in seed_point_ids))

    def test_run_ssdbcodi_uses_bootstrap_when_no_labels(self):
        result = run_ssdbcodi(self.matrix, labeling_state=None, n_clusters=3)

        self.assertIsInstance(result, SsdbcodiResult)
        self.assertTrue(result.parameters["bootstrap_used"])
        self.assertEqual(len(result.point_scores), len(self.matrix.point_ids))
        self.assertGreaterEqual(len(result.outlier_result.outlier_point_ids), 1)
        self.assertEqual(set(result.outlier_result.outlier_point_ids), {
            "outlier_far_01",
            "outlier_far_02",
            "outlier_far_03",
        })
        self.assertEqual(cluster_counts(result), {
            "cluster_1": 6,
            "cluster_2": 6,
            "cluster_3": 6,
        })

        for score in result.point_scores:
            self.assertIsInstance(score, PointScores)
            self.assertTrue(0.0 <= score.r_score <= 1.0)
            self.assertTrue(0.0 <= score.l_score <= 1.0)
            self.assertTrue(0.0 <= score.sim_score <= 1.0)

    def test_run_ssdbcodi_keeps_bootstrap_anchors_when_labels_provided(self):
        labeling_state = LabelingState(
            dataset_id=self.matrix.point_ids[0] and ssdbcodi_dataset_id(),
            annotations=(
                ManualAnnotation(
                    annotation_id="a1",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_a_01",),
                    label_type="cluster",
                    label_value="cluster_alpha",
                ),
                ManualAnnotation(
                    annotation_id="a2",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_b_01",),
                    label_type="cluster",
                    label_value="cluster_beta",
                ),
            ),
        )

        result = run_ssdbcodi(self.matrix, labeling_state=labeling_state)

        self.assertTrue(result.parameters["bootstrap_used"])
        self.assertEqual(result.diagnostics["manual_seed_count"], 2)
        self.assertGreaterEqual(result.diagnostics["bootstrap_seed_count"], 3)
        sources = {seed.source for seed in result.seeds}
        self.assertEqual(sources, {"kmeans_bootstrap", "manual_label"})
        cluster_ids = {assignment.cluster_id for assignment in result.cluster_result.assignments}
        self.assertIn("cluster_alpha", cluster_ids)
        self.assertIn("cluster_beta", cluster_ids)

    def test_manual_relabel_does_not_drop_other_bootstrap_clusters(self):
        labeling_state = LabelingState(
            dataset_id=ssdbcodi_dataset_id(),
            annotations=(
                ManualAnnotation(
                    annotation_id="a1",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_c_04",),
                    label_type="cluster",
                    label_value="cluster_3",
                ),
            ),
        )

        result = run_ssdbcodi(
            self.matrix,
            labeling_state=labeling_state,
            n_clusters=4,
        )

        cluster_ids = {
            assignment.cluster_id
            for assignment in result.cluster_result.assignments
        }
        self.assertIn("cluster_1", cluster_ids)
        self.assertIn("cluster_2", cluster_ids)
        relabeled = next(
            score for score in result.point_scores if score.point_id == "ring_c_04"
        )
        self.assertEqual(relabeled.cluster_id, "cluster_3")

    def test_run_ssdbcodi_persists_intermediate_scores(self):
        result = run_ssdbcodi(self.matrix, n_clusters=3)

        for score in result.point_scores:
            self.assertIsNotNone(score.r_score)
            self.assertIsNotNone(score.l_score)
            self.assertIsNotNone(score.sim_score)
            self.assertIsNotNone(score.t_score)
            self.assertIsNotNone(score.c_dist)
            self.assertIsNotNone(score.e_max)

    def test_outlier_label_overrides_force_outlier_state(self):
        labeling_state = LabelingState(
            dataset_id=ssdbcodi_dataset_id(),
            annotations=(
                ManualAnnotation(
                    annotation_id="a1",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_a_01",),
                    label_type="cluster",
                    label_value="cluster_alpha",
                ),
                ManualAnnotation(
                    annotation_id="a2",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_b_01",),
                    label_type="cluster",
                    label_value="cluster_beta",
                ),
                ManualAnnotation(
                    annotation_id="a3",
                    dataset_id=ssdbcodi_dataset_id(),
                    source="manual_label",
                    scope="selected_points",
                    point_ids=("ring_a_02",),
                    label_type="outlier",
                    label_value=True,
                ),
            ),
        )

        result = run_ssdbcodi(self.matrix, labeling_state=labeling_state)

        outlier_ids = set(result.outlier_result.outlier_point_ids)
        self.assertIn("ring_a_02", outlier_ids)
        labeled_score = next(
            score for score in result.point_scores if score.point_id == "ring_a_02"
        )
        self.assertAlmostEqual(labeled_score.sim_score, 1.0)

    def test_seed_records_carry_source(self):
        result = run_ssdbcodi(self.matrix, n_clusters=3)

        sources = {seed.source for seed in result.seeds}
        self.assertEqual(sources, {"kmeans_bootstrap"})
        for seed in result.seeds:
            self.assertIsInstance(seed, SeedRecord)

    def test_store_records_history_for_rerun(self):
        store = SsdbcodiStore(dataset_id=ssdbcodi_dataset_id())
        first = run_ssdbcodi(self.matrix, n_clusters=3)
        second = run_ssdbcodi(self.matrix, n_clusters=2)
        store.record_result(first)
        store.record_result(second)

        self.assertEqual(len(store.history), 2)
        self.assertIs(store.latest_result, second)
        summary = store.history_summary()
        self.assertEqual(len(summary), 2)
        self.assertIn("run_id", summary[0])

    def test_collect_seeds_from_labeling_handles_none(self):
        seeds, overrides = collect_seeds_from_labeling(self.matrix, None)
        self.assertEqual(seeds, {})
        self.assertEqual(overrides, {})

    def test_run_ssdbcodi_supports_adjustable_bootstrap_k(self):
        two = run_ssdbcodi(self.matrix, n_clusters=2)
        four = run_ssdbcodi(self.matrix, n_clusters=4)

        two_clusters = {assignment.cluster_id for assignment in two.cluster_result.assignments}
        four_clusters = {assignment.cluster_id for assignment in four.cluster_result.assignments}

        self.assertGreaterEqual(len(four_clusters), len(two_clusters))

    def test_cluster_counts_returns_per_cluster_totals(self):
        result = run_ssdbcodi(self.matrix, n_clusters=3)
        counts = cluster_counts(result)

        self.assertEqual(
            sum(counts.values()),
            len(result.cluster_result.assignments),
        )


if __name__ == "__main__":
    unittest.main()
