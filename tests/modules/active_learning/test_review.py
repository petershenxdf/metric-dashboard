import math
import unittest
from dataclasses import replace

import numpy as np

from app.modules.active_learning.review import (
    CATEGORY_IDS,
    build_review,
    choose_batch,
    instability_scores,
    midrank_percentiles,
)
from app.modules.active_learning.schemas import LabelEvent
from app.modules.ssdbcodi.algorithm import (
    core_distances,
    expansion_reachability,
    expansion_tree,
    pairwise_euclidean,
    reachability_matrix,
    run_ssdbcodi_core,
)
from app.shared.schemas import (
    AnalysisResult,
    ClusterAssignment,
    ClusterResult,
    FeatureMatrix,
    OutlierResult,
    OutlierScore,
)


def analysis_for(matrix, groups, outliers=()):
    distance = pairwise_euclidean(np.asarray(matrix.values))
    tree = expansion_tree(reachability_matrix(distance, core_distances(distance, 1)))
    return AnalysisResult(
        analysis_run_id="test_analysis",
        cluster_result=ClusterResult(
            "test_cluster",
            "test",
            max(1, len(set(groups.values()))),
            tuple(
                ClusterAssignment(pid, group)
                for pid, group in groups.items()
                if pid not in outliers
            ),
        ),
        outlier_result=OutlierResult(
            "test_outlier",
            "test",
            tuple(OutlierScore(pid, 0.0, pid in outliers) for pid in matrix.point_ids),
        ),
        diagnostics={
            "expansion_tree": [
                {"from": matrix.point_ids[a], "to": matrix.point_ids[b], "distance": d}
                for a, b, d in tree
            ]
        },
    )


def label(pid, dimension, value):
    return LabelEvent(
        "event_" + pid + dimension,
        "session",
        "round",
        pid,
        dimension,
        value,
        "active",
        None,
        {},
        "2026-09-11T00:00:00Z",
    )


class SixScoreTests(unittest.TestCase):
    def setUp(self):
        self.matrix = FeatureMatrix(
            tuple("abcdef"), ("x",), ((0.0,), (0.2,), (0.4,), (1.0,), (2.0,), (10.0,))
        )
        self.analysis = analysis_for(
            self.matrix, dict(zip("abcdef", ("A", "B", "B", "B", "B", "C")))
        )

    def review(self, events=(), **options):
        return build_review(
            self.matrix,
            self.analysis,
            events,
            {1: self.analysis, 2: self.analysis},
            min_pts=1,
            **options,
        )

    def score(self, review, pid, category):
        return next(row for row in review["rows"] if row["point_id"] == pid)["scores"][
            CATEGORY_IDS[category - 1]
        ]

    def test_midrank_document_example_and_ties(self):
        values = {f"p{i}": i for i in range(100)}
        for i in range(80, 84):
            values[f"p{i}"] = 80
        result = midrank_percentiles(values)
        self.assertEqual(result["p80"], 82)
        self.assertEqual(result["p83"], 82)
        self.assertEqual(midrank_percentiles({"a": 1, "b": 1}), {"a": 50, "b": 50})
        self.assertEqual(midrank_percentiles({}), {})

    def test_conflict_uses_normal_neighbors_and_skips_outliers(self):
        result = self.review()
        score = self.score(result, "a", 1)
        self.assertEqual(score["raw"], 1)
        self.assertEqual(score["comparison_point_ids"], ["b"])
        self.analysis = analysis_for(
            self.matrix, {pid: "A" for pid in "abcdef"}, outliers=("a",)
        )
        self.assertIsNone(self.score(self.review(), "a", 1)["raw"])
        self.assertNotIn("a", self.score(self.review(), "b", 1)["comparison_point_ids"])

    def test_conflict_zero_does_not_recommend_even_when_percentile_high(self):
        self.analysis = analysis_for(self.matrix, {pid: "A" for pid in "abcdef"})
        score = self.score(self.review(threshold=0), "a", 1)
        self.assertEqual(score["raw"], 0)
        self.assertFalse(score["recommendable"])

    def test_missing_human_references_are_na_not_zero(self):
        result = self.review()
        for category in (2, 3, 5):
            self.assertIsNone(self.score(result, "a", category)["raw"])
            self.assertIsNone(self.score(result, "a", category)["percentile"])

    def test_coverage_uses_semantic_only_and_excludes_self(self):
        result = self.review([label("a", "semantic_class", "type")])
        self.assertIsNone(self.score(result, "a", 2)["raw"])
        self.assertAlmostEqual(self.score(result, "b", 2)["raw"], 0.2)
        normal_only = self.review([label("a", "outlier_status", False)])
        self.assertIsNone(self.score(normal_only, "b", 2)["raw"])
        self.assertIsNotNone(self.score(normal_only, "b", 3)["raw"])

    def test_coverage_support_uses_nearest_rank_90_percent_and_all_data(self):
        self.matrix = FeatureMatrix(
            tuple(f"p{i}" for i in range(11)),
            ("x",),
            tuple((float(i),) for i in range(10)) + ((100.0,),),
        )
        self.analysis = analysis_for(
            self.matrix, {pid: "A" for pid in self.matrix.point_ids}
        )
        result = self.review([label("p0", "semantic_class", "type")])
        self.assertIsNone(self.score(result, "p10", 2)["raw"])
        self.assertEqual(self.score(result, "p9", 2)["evidence"]["support_cutoff"], 1)
        self.assertEqual(result["categories"][1]["pool_size"], 9)

    def test_expansion_uses_full_path_not_direct_reference_distance(self):
        result = self.review([label("a", "outlier_status", False)])
        # a -> b -> c -> d -> e has maximum edge 1, not direct distance 2.
        self.assertAlmostEqual(self.score(result, "e", 3)["raw"], 1 - math.exp(-1))
        self.assertEqual(
            self.score(result, "e", 3)["evidence"]["normal_reference"], "a"
        )
        self.assertIsNone(self.score(result, "a", 3)["raw"])
        core = run_ssdbcodi_core(self.matrix.values, {0: "A"}, min_pts=1)
        self.assertAlmostEqual(core["e_max"][4], 1)
        self.assertAlmostEqual(core["r_score"][4], math.exp(-1))

    def test_zero_distance_expansion_edges_are_retained(self):
        edges = expansion_tree(np.zeros((3, 3)))
        self.assertEqual(len(edges), 2)
        barriers, origins = expansion_reachability(3, edges, [0], exclude_self=True)
        self.assertTrue(math.isinf(barriers[0]))
        self.assertEqual(list(barriers[1:]), [0, 0])

    def test_incomplete_expansion_is_na(self):
        self.analysis = replace(self.analysis, diagnostics={})
        self.assertIsNone(
            self.score(self.review([label("a", "semantic_class", "type")]), "b", 3)[
                "raw"
            ]
        )

    def test_sparsity_uses_all_neighbors(self):
        self.analysis = analysis_for(
            self.matrix, {pid: "A" for pid in "abcdef"}, outliers=("b",)
        )
        score = self.score(self.review(), "a", 4)
        self.assertEqual(score["comparison_point_ids"], ["b"])
        self.assertAlmostEqual(score["raw"], 1 - math.exp(-0.2))

    def test_sparsity_insufficient_neighbors_is_na(self):
        result = build_review(
            self.matrix, self.analysis, (), {1: self.analysis}, min_pts=6
        )
        self.assertIsNone(self.score(result, "a", 4)["raw"])

    def test_known_outlier_raw_score_survives_failed_local_match(self):
        result = self.review([label("a", "outlier_status", True)], threshold=0)
        close, far = self.score(result, "b", 5), self.score(result, "f", 5)
        self.assertAlmostEqual(close["raw"], math.exp(-0.2))
        self.assertTrue(close["recommendable"])
        self.assertAlmostEqual(far["raw"], math.exp(-10))
        self.assertIsNotNone(far["percentile"])
        self.assertFalse(far["recommendable"])
        self.assertIn("local-match", far["ineligibility_reason"])
        self.assertIsNone(self.score(result, "a", 5)["raw"])

    def test_unsure_and_predicted_outliers_are_not_references(self):
        self.analysis = analysis_for(
            self.matrix, {pid: "A" for pid in "abcdef"}, outliers=("a",)
        )
        result = self.review([label("a", "uncertain", True)])
        for category in (2, 3, 5):
            self.assertIsNone(self.score(result, "b", category)["raw"])

    def test_instability_invariant_to_names_and_document_flip_example(self):
        base = analysis_for(
            self.matrix, {pid: "A" if pid in "abc" else "B" for pid in "abcdef"}
        )
        renamed = analysis_for(
            self.matrix, {pid: "X" if pid in "abc" else "Y" for pid in "abcdef"}
        )
        flipped = analysis_for(
            self.matrix,
            {pid: "X" if pid in "abc" else "Y" for pid in "abcdef"},
            outliers=("a", "b", "c"),
        )
        self.assertEqual(
            instability_scores([base, renamed], self.matrix.point_ids)["a"], 0
        )
        self.assertAlmostEqual(
            instability_scores([base, renamed, flipped], self.matrix.point_ids)["a"],
            2 / 3,
        )
        self.assertEqual(
            instability_scores([flipped, flipped], self.matrix.point_ids)["a"], 0
        )

    def test_instability_detects_splits_merges_and_missing_results(self):
        one = analysis_for(
            self.matrix, {pid: "A" if pid in "abc" else "B" for pid in "abcdef"}
        )
        two = analysis_for(
            self.matrix, {pid: "A" if pid in "abcde" else "B" for pid in "abcdef"}
        )
        self.assertAlmostEqual(
            instability_scores([one, two], self.matrix.point_ids)["a"], 0.4
        )
        incomplete = analysis_for(self.matrix, {pid: "A" for pid in "abcde"})
        self.assertEqual(
            instability_scores([one, incomplete], self.matrix.point_ids), {}
        )
        self.assertEqual(instability_scores([one], self.matrix.point_ids), {})

    def test_failed_rerun_makes_instability_na(self):
        result = self.review(run_errors={"2": "failed"})
        self.assertIsNone(self.score(result, "a", 6)["raw"])

    def test_eligible_pool_excludes_labeled_and_deferred(self):
        result = self.review(
            statuses={"a": "labeled", "b": "deferred", "c": "re_review"}
        )
        self.assertEqual(result["categories"][3]["pool_size"], 4)
        self.assertIsNone(self.score(result, "a", 4)["percentile"])
        self.assertIsNotNone(self.score(result, "c", 4)["percentile"])

    def test_batch_deduplicates_and_uses_id_ties(self):
        def row(pid, duplicate):
            return {
                "point_id": pid,
                "duplicate_key": duplicate,
                "scores": {"category": {"percentile": 90, "recommendable": True}},
            }

        self.assertEqual(
            choose_batch([row("b", 1), row("a", 1), row("c", 2)], "category", 4),
            ["a", "c"],
        )

    def test_budget_does_not_erase_scores(self):
        result = self.review(budget_reached=True)
        self.assertIsNotNone(self.score(result, "a", 4)["percentile"])
        self.assertFalse(any(c["recommended_point_ids"] for c in result["categories"]))


if __name__ == "__main__":
    unittest.main()
