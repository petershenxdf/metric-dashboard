from __future__ import annotations

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_feature_matrix
from app.modules.labeling.service import get_labeling_state
from app.modules.labeling.state import get_debug_store_for_context
from app.modules.projection.service import project_feature_matrix
from app.modules.selection.service import get_selection_context, get_selection_state, list_selection_groups
from app.modules.selection.state import get_debug_store_for_dataset
from app.shared.effective_analysis import apply_manual_labels_to_analysis
from app.shared.fixtures import (
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
)

from .service import build_render_payload


def scatterplot_fixture_state(n_clusters: int = DEFAULT_N_CLUSTERS):
    dataset = analysis_selection_dataset(DEFAULT_WORKFLOW_DATASET_ID)
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    selection_store = get_debug_store_for_dataset(
        dataset,
        analysis_selection_initial_selected_point_ids(dataset.dataset_id),
    )
    selection_state = get_selection_state(selection_store)
    context = get_selection_context(selection_store)
    labeling_state = get_labeling_state(get_debug_store_for_context(context))
    provider_labeling_state = labeling_state if labeling_state.annotations else None
    provider_analysis = run_default_analysis(
        matrix,
        n_clusters=n_clusters,
        labeling_state=provider_labeling_state,
    )
    analysis = apply_manual_labels_to_analysis(dataset, provider_analysis, labeling_state)
    render_payload = build_render_payload(
        dataset=dataset,
        projection=projection,
        clusters=analysis.cluster_result,
        outliers=analysis.outlier_result,
        selection_context=context,
        labeling_state=labeling_state,
    )

    return {
        "dataset": dataset,
        "matrix": matrix,
        "projection": projection,
        "provider_analysis": provider_analysis,
        "analysis": analysis,
        "selection_store": selection_store,
        "selection_state": selection_state,
        "selection_groups": list_selection_groups(selection_store),
        "context": context,
        "labeling_state": labeling_state,
        "render_payload": render_payload,
        "n_clusters": n_clusters,
    }
