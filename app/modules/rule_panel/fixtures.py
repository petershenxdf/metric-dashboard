from __future__ import annotations

from app.modules.algorithm_adapters.service import (
    DEFAULT_N_CLUSTERS,
    cluster_counts,
    run_default_analysis,
)
from app.shared.wine_dataset import WINE_DATASET_ID, load_wine_dataset, load_wine_feature_matrix

from .schemas import TreeConfig
from .service import generate_rule_set, interpret_rule_set_preview


RULE_PANEL_DATASET_ID = WINE_DATASET_ID


def rule_panel_fixture_state(
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    max_depth: int = 3,
    min_samples_leaf: int = 1,
):
    dataset = load_wine_dataset()
    feature_matrix = load_wine_feature_matrix()
    analysis = run_default_analysis(feature_matrix, n_clusters=n_clusters)
    config = TreeConfig(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    rule_set = generate_rule_set(
        feature_matrix,
        analysis,
        dataset_id=dataset.dataset_id,
        config=config,
    )
    interpretation = interpret_rule_set_preview(rule_set)

    return {
        "dataset": dataset,
        "feature_matrix": feature_matrix,
        "analysis": analysis,
        "cluster_counts": cluster_counts(analysis.cluster_result),
        "tree_config": config,
        "rule_set": rule_set,
        "interpretation": interpretation,
    }
