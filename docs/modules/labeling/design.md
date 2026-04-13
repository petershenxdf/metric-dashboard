# Labeling Module Design

## Purpose

The labeling module owns manual point annotations.

It turns selected point IDs into explicit user supervision, such as "these selected points are cluster 2" or "these selected points are outliers".

This module is the direct UI path for feedback. Chat-based feedback can later produce the same structured instruction family through the intent instruction module.

## Responsibilities

1. Receive selection context from the selection module.
2. Validate selected and target point IDs.
3. Create manual annotations for cluster/class labels.
4. Create manual annotations for outlier and not-outlier labels.
5. Keep an annotation history for local debugging.
6. Output structured feedback instructions for downstream metric learning.
7. Expose labeling state and actions through Flask.

## Not Responsible For

1. Owning selected point state.
2. Rendering scatterplot geometry.
3. Parsing natural language.
4. Running clustering or outlier detection.
5. Running metric learning.
6. Deciding final algorithm output.

## Target Files

```text
app/modules/labeling/
  __init__.py
  schemas.py
  service.py
  fixtures.py
  routes.py
  templates/labeling/index.html
  static/labeling/labeling.js

tests/modules/labeling/
  test_service.py
  test_routes.py
```

## Annotation Contract

```json
{
  "annotation_id": "annotation_001",
  "dataset_id": "iris_sample",
  "source": "manual_label",
  "scope": "selected_points",
  "point_ids": ["p1", "p7", "p9"],
  "label_type": "cluster",
  "label_value": "cluster_2",
  "status": "active"
}
```

Outlier annotation:

```json
{
  "annotation_id": "annotation_002",
  "dataset_id": "iris_sample",
  "source": "manual_label",
  "scope": "selected_points",
  "point_ids": ["p88", "p102"],
  "label_type": "outlier",
  "label_value": true,
  "status": "active"
}
```

## Structured Feedback Output

Manual annotations should produce the same downstream shape that chat-derived instructions use.

Cluster assignment:

```json
{
  "instruction_type": "assign_cluster",
  "status": "actionable",
  "source": "manual_label",
  "target": {
    "source": "selected_points",
    "point_ids": ["p1", "p7", "p9"]
  },
  "parameters": {
    "target_type": "cluster",
    "target_label": "cluster_2"
  }
}
```

Outlier label:

```json
{
  "instruction_type": "is_outlier",
  "status": "actionable",
  "source": "manual_label",
  "target": {
    "source": "selected_points",
    "point_ids": ["p88", "p102"]
  },
  "parameters": {
    "target_type": "outlier"
  }
}
```

## Public Service API

```python
create_cluster_annotation(selection_context, target_label)
create_new_class_annotation(selection_context, class_name)
create_outlier_annotation(selection_context)
create_not_outlier_annotation(selection_context)
list_annotations(dataset_id)
clear_annotations(dataset_id)
annotation_to_instruction(annotation)
```

## Flask Routes

```text
/modules/labeling/                       labeling debug page
/modules/labeling/health                 module health
/modules/labeling/api/state              annotation state JSON
/modules/labeling/api/apply              create annotation from selected points
/modules/labeling/api/annotations        list annotations
/modules/labeling/api/reset              clear local annotations
/workflows/selection-labeling/           selection plus labeling workflow
```

## Flask Debug Page Requirements

The page should show:

1. current selected and unselected point IDs.
2. controls for assigning selected points to an existing cluster.
3. controls for creating a new class label.
4. controls for marking selected points as outliers or not outliers.
5. annotation history.
6. structured feedback JSON preview.
7. clear dependency mode: mock selection or real selection.

## Interaction Rules

1. Selection owns which points are selected.
2. Labeling owns what those selected points mean.
3. Scatterplot may expose label buttons, but it must call labeling APIs.
4. Chatbox may express label intent, but intent instruction should compile it into the same structured feedback shape.
5. Metric-learning adapter should consume the structured feedback, not raw labeling UI state.

## Testing

Unit tests:

1. selected points can become an `assign_cluster` instruction.
2. selected points can become an `is_outlier` instruction.
3. empty selection is rejected.
4. unknown point IDs are rejected.
5. annotation history preserves point IDs and label values.
6. reset clears local annotation state.

Flask route tests:

1. debug page returns 200.
2. state API returns annotation history.
3. apply API creates a valid annotation.
4. reset API clears annotations.

Manual browser check:

1. open `/modules/labeling/`.
2. use fixture selected points.
3. assign them to a cluster.
4. mark a second selection as outliers.
5. confirm annotation history and structured feedback JSON update.
6. open `/workflows/selection-labeling/` to check interaction with real selection when available.

## Completion Criteria

This module is complete when manual labels can be created from selected points, inspected in Flask, and converted into structured feedback without depending on chatbox or metric learning.
