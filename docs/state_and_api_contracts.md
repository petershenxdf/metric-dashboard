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
    projection = None
    cluster_result = None
    outlier_result = None
    selection = None
    annotations = []
    chat_history = []
    structured_instructions = []
    constraints = []
    refinement_runs = []
```

This can start as a simple object or dictionary attached to `app.config` or a small state module.

## 3. State Ownership

Each state area has one owner:

| State | Owner |
| --- | --- |
| dataset and feature matrix | `data_workspace` |
| projection coordinates | `projection` |
| cluster assignments | `algorithm_adapters` |
| outlier scores | `algorithm_adapters` |
| selected point IDs | `selection` |
| manual cluster/outlier annotations | `labeling` |
| chat history | `chatbox` |
| structured instructions | `intent_instruction` |
| metric constraints | `metric_learning_adapter` |
| refinement run history | `refinement_orchestrator` |

Other modules may read state through contracts, but should not mutate state they do not own.

Structured feedback can originate from two modules:

1. `labeling` for direct UI actions such as assigning selected points to a cluster or marking outliers.
2. `intent_instruction` for chat-derived feedback.

Both sources should use the same instruction shape before reaching `metric_learning_adapter`.

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
/modules/selection/api/state
/modules/labeling/api/state
/modules/chatbox/api/context
```

Interactive modules should expose action APIs:

```text
/modules/selection/api/select
/modules/labeling/api/apply
/modules/chatbox/api/messages
/modules/intent-instruction/api/compile
```

Labeling action payloads should use selected point IDs and produce structured feedback:

```json
{
  "action": "assign_cluster",
  "scope": "selected_points",
  "point_ids": ["p1", "p7", "p9"],
  "target_label": "cluster_2"
}
```

## 7. Reset Rule

For local debugging, stateful modules should support a reset path when useful:

```text
/modules/<module>/api/reset
```

This is especially useful for:

1. selection.
2. labeling.
3. chatbox.
4. refinement orchestrator.

## 8. Fixture Rule

Every module should have fixtures that make the module page useful without the full app.

A fixture should specify:

1. what is real.
2. what is mocked.
3. what previous module output it imitates.

Example:

```json
{
  "fixture_name": "iris_projection_debug",
  "real_inputs": ["data_workspace", "projection"],
  "mocked_inputs": ["cluster_result", "outlier_result"]
}
```
