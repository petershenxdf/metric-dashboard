# Local State and API Contracts

## 1. Purpose

This document defines how modules should share state and shape API responses in the local Flask app.

The goal is simple local development, not production infrastructure.

## 2. Local State Rule

Use in-memory state during early development.

Do not add a production database early.

Suggested app state:
```python
class AppState:
    dataset = None
    feature_matrix = None
    transformed_feature_matrix = None  # X @ L (Path A) or X @ S (Path B)
    projection = None
    cluster_result = None
    outlier_result = None
    ssdbcodi_result = None               # SsdbcodiResult with per-point scores
    selection = None
    selection_groups = []
    annotations = []
    chat_history = []
    active_refinement_strategy = None  # "metric_learning" | "direct_ssdbcodi"
    structured_instruction = None      # evolving single state (not a list)

    # Path A (metric learning)
    active_learned_metric = None       # {M, L, provider, diagnostics}
    metric_refinement_runs = []        # Path A history for rollback

    # Path B (direct SSDBCODI)
    active_direct_feedback_plan = None # {seed_updates, feature_scale, param_overrides, ...}
    direct_refinement_runs = []        # Path B history for rollback
```

This can start as a simple object or dictionary attached to `app.config` or a small state module.

## 3. State Ownership

Each state area has one owner:

| State | Owner |
| --- | --- |
| dataset and feature matrix | `data_workspace` |
| projection coordinates | `projection` |
| cluster assignments | `algorithm_adapters` backed by SSDBCODI; `ssdbcodi` debug store for standalone runs |
| outlier scores | `algorithm_adapters` backed by SSDBCODI `tScore`; `ssdbcodi` debug store for standalone runs |
| SSDBCODI intermediate scores (`rScore`, `lScore`, `simScore`, `tScore`) | `ssdbcodi` |
| selected point IDs | `selection` |
| manual cluster/outlier annotations | `labeling` |
| chat history and active refinement strategy | `chatbox` |
| structured instruction state | `intent_instruction` |
| Path A metric constraint set and learned metric | `metric_learning_adapter` |
| Path B direct feedback plan | `direct_feedback_adapter` |
| Path A refinement run history and active metric pointer | `metric_refinement_orchestrator` |
| Path B refinement run history and active plan pointer | `direct_refinement_orchestrator` |

Other modules may read state through contracts, but should not mutate state they do not own.

SSDBCODI follows the same ownership split on its debug page and when used
through `algorithm_adapters`: selection remains owned by `selection`, manual
labels remain owned by `labeling`, and `ssdbcodi` owns only its computed
result/history/scores. `POST
/modules/ssdbcodi/api/label` records pending labeling feedback; `POST
/modules/ssdbcodi/api/run` is the explicit boundary that recomputes and stores
SSDBCODI output. These states are scoped by `dataset_id` for the module's
debug fixtures.

Structured feedback can originate from two modules:

1. `labeling` for direct UI actions such as assigning selected points to a cluster or marking outliers.
2. `intent_instruction` for chat-derived feedback. This module owns a single evolving `StructuredInstruction` state, updated turn by turn through deltas rather than regenerated from scratch.

These two sources feed two parallel adapters:

1. **Path A** — `metric_learning_adapter.constraint_builder` merges them into a `ConstraintSet` and passes it to the metric learner. The adapter's learned matrix `L = chol(M)` is applied as a linear pre-transform to the feature matrix so projection and algorithm adapters can be reused without modification.
2. **Path B** — `direct_feedback_adapter.plan_builder` merges them into a `DirectFeedbackPlan` (seed updates, `feature_scale`, `param_overrides`, `excluded_clusters`, `merged_cluster_groups`) that is fed directly to SSDBCODI through `algorithm_adapters.run_default_analysis`.

Phase 1 intents:

- Shared on both paths: `feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`.
- Path B-only: `split_cluster` (realized as `n_clusters += 1` plus interior seeds) and `reclassify_outlier` (realized as a labeled outlier override). Path A rejects these with `intent_deferred` and a `suggested_strategy: "direct_ssdbcodi"` hint because a learned metric cannot change KMeans's `k` and may not move a point across SSDBCODI's contamination threshold.

## 4. API Response Envelope

Use a consistent JSON response shape for debug APIs:

```json
{
  "ok": true,
  "data": {},
  "error": null,
  "diagnostics": {}
}
```

For errors:

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "invalid_input",
    "message": "selected point id is unknown"
  },
  "diagnostics": {}
}
```

This makes module pages and workflow pages easier to debug.

## 5. Route Naming Convention

Python package names should use snake_case:

```text
data_workspace
intent_instruction
metric_learning_adapter
labeling
```

Flask route slugs should use kebab-case:

```text
data-workspace
intent-instruction
metric-learning-adapter
labeling
```

The module registry should define the mapping explicitly.

## 6. Required Module APIs

Each module should expose:

```text
/modules/<module>/health
```

Each module should expose at least one state or primary data API, such as:

```text
/modules/data-workspace/api/dataset
/modules/projection/api/projection
/modules/algorithm-adapters/api/outliers
/modules/algorithm-adapters/api/clusters
/modules/algorithm-adapters/api/analysis
/modules/ssdbcodi/api/state
/modules/ssdbcodi/api/scores
/modules/ssdbcodi/api/result
/modules/selection/api/state
/modules/selection/api/context
/modules/selection/api/groups
/modules/labeling/api/state
/modules/chatbox/api/context
```

Interactive modules should expose action APIs:

```text
/modules/selection/api/select
/modules/selection/api/deselect
/modules/selection/api/replace
/modules/selection/api/toggle
/modules/selection/api/clear
/modules/selection/api/groups
/modules/selection/api/groups/<id>/select
/modules/labeling/api/apply
/modules/ssdbcodi/api/run
/modules/ssdbcodi/api/select
/modules/ssdbcodi/api/groups
/modules/ssdbcodi/api/label
/modules/ssdbcodi/api/clear-labels
/workflows/analysis-labeling/api/select
/workflows/analysis-labeling/api/label
/workflows/analysis-labeling/api/clear-labels
/modules/scatterplot/api/render-payload
/modules/scatterplot/api/select
/modules/scatterplot/api/toggle
/modules/scatterplot/api/groups
/workflows/scatter-selection/api/state
/workflows/scatter-selection/api/select
/workflows/scatter-selection/api/groups
/workflows/scatter-labeling/api/state
/workflows/scatter-labeling/api/select
/workflows/scatter-labeling/api/label
/workflows/scatter-labeling/api/groups
/workflows/provider-feedback/api/state
/modules/chatbox/api/messages
/modules/chatbox/api/history
/modules/intent-instruction/api/route
/modules/intent-instruction/api/compile
/modules/intent-instruction/api/state
/modules/metric-learning-adapter/api/constraints
/modules/metric-learning-adapter/api/fit
/modules/metric-learning-adapter/api/providers
/modules/direct-feedback-adapter/api/plan
/modules/direct-feedback-adapter/api/preview
/modules/metric-refinement-orchestrator/api/run
/modules/metric-refinement-orchestrator/api/history
/modules/metric-refinement-orchestrator/api/rollback
/modules/metric-refinement-orchestrator/api/reset
/modules/direct-refinement-orchestrator/api/run
/modules/direct-refinement-orchestrator/api/history
/modules/direct-refinement-orchestrator/api/rollback
/modules/direct-refinement-orchestrator/api/reset
/workflows/instruction-constraints/api/state
/workflows/instruction-ssdbcodi/api/state
/workflows/metric-refinement-loop/api/state
/workflows/direct-refinement-loop/api/state
/workflows/strategy-comparison/api/state
/workflows/strategy-comparison/api/run
```

Every module-level state route should include module identity fields in the
standard JSON envelope:

```json
{
  "ok": true,
  "data": {
    "module": "selection",
    "status": "working"
  }
}
```

Module-specific state fields should live beside `module` and `status`, not in a
separate nested object, unless that module already documents a nested contract.

Selection action payloads should be action-route based and extensible:

```json
{
  "point_ids": ["setosa_001", "versicolor_001"],
  "source": "lasso",
  "mode": "replace",
  "metadata": {
    "gesture_id": "lasso_001"
  }
}
```

Supported selection actions:

```text
select
deselect
replace
toggle
clear
```

Supported selection sources should start with:

```text
api
point_click
lasso
rectangle
manual_list
workflow_fixture
selection_group
```

Future selection gestures should add new source or mode values while preserving
the same selected/unselected context output shape.

Selection groups are reusable named point sets owned by the selection module.
They are useful for restoring a previous selection, but they are not semantic
labels or metric-learning constraints.

```json
{
  "group_id": "group_001",
  "group_name": "interesting pair",
  "dataset_id": "selection_iris_debug",
  "point_ids": ["setosa_001", "versicolor_001"],
  "point_count": 2,
  "metadata": {}
}
```

Group routes:

```text
GET    /modules/selection/api/groups
POST   /modules/selection/api/groups
POST   /modules/selection/api/groups/<id>/select
DELETE /modules/selection/api/groups/<id>
```

Creating a group without `point_ids` saves the current active selection.
Selecting a group applies `replace` selection with `source: "selection_group"`.

Labeling action payloads should use selected point IDs and produce structured feedback:

```json
{
  "action": "assign_cluster",
  "scope": "selected_points",
  "point_ids": ["p1", "p7", "p9"],
  "target_label": "cluster_2"
}
```

Current labeling apply payload:

```json
{
  "action": "assign_cluster",
  "label_value": "cluster_2",
  "point_ids": ["p1", "p7"]
}
```

If `point_ids` is omitted, labeling applies to all current selected points.
Explicit `point_ids` must be selected in the current selection context.

Supported labeling actions:

```text
assign_cluster
assign_new_class
mark_outlier
mark_not_outlier
```

The Step 1-5 workflow exposes a combined state payload at:

```text
/workflows/analysis-labeling/api/state
```

That payload includes dataset, feature matrix, projection, outliers, clusters,
selection state, selection context, selection groups, and labeling state.

For `/workflows/analysis-labeling/`, `clusters` and `outliers` are the final
display state after manual labels are passed into SSDBCODI and explicit label
overrides are applied for UI consistency. Baseline provider outputs remain
available as `raw_clusters` and `raw_outliers`; label-aware provider outputs
remain available as `provider_clusters` and `provider_outliers`.

Allowed labels in this workflow are:

```text
cluster_1
cluster_2
cluster_3
...
outlier
```

The available cluster labels are determined by the current `n_clusters` value.
`cluster_N` makes the target points non-outliers in effective state. `outlier`
removes the target points from effective cluster assignments.

The Step 1-6 scatterplot workflows expose render state at:

```text
/modules/scatterplot/api/render-payload
/workflows/scatter-selection/api/state
/workflows/scatter-labeling/api/state
```

The render payload is derived state, not new ownership. Dataset, projection,
cluster, outlier, selection, and label truth remain owned by their original
modules. Scatterplot points include:

```json
{
  "point_id": "p1",
  "x": 0.2,
  "y": -0.7,
  "screen_x": 240.0,
  "screen_y": 300.0,
  "cluster_id": "cluster_1",
  "is_outlier": false,
  "selected": true,
  "manual_labels": [],
  "metadata": {},
  "color": "#2f6fed"
}
```

The Step 6.5 provider diagnostics workflow exposes:

```text
/workflows/provider-feedback/api/state
```

That payload includes the adapter-facing `AnalysisResult`, standalone
`SsdbcodiResult`, cluster counts for both views, and the active provider name.

Scatterplot selection actions must preserve the same selection action contract
as the selection module, including `source: "point_click"` and
`source: "rectangle"`. Saved selection groups in scatterplot workflows are the
same selection-module groups; they are not labels or constraints.

Algorithm adapter APIs should expose point-ID-based outputs.

Outlier result:

```json
{
  "outlier_run_id": "outlier_001",
  "algorithm": "local_outlier_factor_numpy",
  "scores": [
    {
      "point_id": "p1",
      "score": 1.42,
      "is_outlier": true
    }
  ],
  "outlier_point_ids": ["p1"],
  "diagnostics": {}
}
```

Cluster result:

```json
{
  "cluster_run_id": "cluster_001",
  "algorithm": "kmeans_numpy_deterministic",
  "n_clusters": 3,
  "assignments": [
    {
      "point_id": "p2",
      "cluster_id": "cluster_1"
    }
  ],
  "excluded_outlier_point_ids": ["p1"],
  "diagnostics": {}
}
```

The current default analysis order is:

```text
local_outlier_factor -> kmeans_on_non_outliers
```

Future integrated algorithms should return the same dashboard-facing result shape,
even if they compute clusters and outliers in one combined pass.

## 7. Reset Rule

For local debugging, stateful modules should support a reset path when useful:

```text
/modules/<module>/api/reset
```

This is especially useful for:

1. selection.
2. labeling.
3. chatbox.
4. `metric_refinement_orchestrator` (Path A history only).
5. `direct_refinement_orchestrator` (Path B history only).

Each orchestrator's reset endpoint scopes to its own history so resetting one path does not wipe the other.

## 8. Fixture Rule

Every module should have fixtures that make the module page useful without the full app.

A fixture should specify:

1. what is real.
2. what is mocked.
3. what previous module output it imitates.

Example:

```json
{
  "fixture_name": "default_analysis_outlier_debug",
  "real_inputs": ["data_workspace", "projection", "algorithm_adapters"],
  "mocked_inputs": []
}
```
