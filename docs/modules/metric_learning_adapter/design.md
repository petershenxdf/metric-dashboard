# Metric-Learning Adapter Module Design

## Purpose

The metric-learning adapter converts structured instructions into constraint payloads for the metric-learning pipeline.

It should not receive raw chat text. It should only receive validated structured instructions.

## Responsibilities

1. Accept structured instructions.
2. Validate instruction status.
3. Reject incomplete and non-actionable instructions.
4. Convert instructions into metric-learning constraints.
5. Call the metric-learning pipeline when available.
6. Provide a Flask page for constraint preview.

## Not Responsible For

1. Parsing natural language.
2. Rendering chat UI.
3. Managing selection state.
4. Running clustering directly.
5. Running outlier detection directly.

## Target Files

```text
app/modules/metric_learning_adapter/
  __init__.py
  schemas.py
  adapter.py
  fixtures.py
  routes.py
  templates/metric_learning_adapter/index.html

tests/modules/metric_learning_adapter/
  test_adapter.py
  test_routes.py
```

## Input Contract

```json
{
  "instruction_type": "same_class",
  "status": "actionable",
  "target": {
    "source": "selected_points",
    "point_ids": ["p1", "p7", "p9"]
  },
  "parameters": {}
}
```

## Constraint Examples

Same class:

```json
{
  "constraint_type": "must_link",
  "point_ids": ["p1", "p7", "p9"]
}
```

Different class:

```json
{
  "constraint_type": "cannot_link",
  "groups": [["p1"], ["p7"]]
}
```

Split:

```json
{
  "constraint_type": "split",
  "point_ids": ["p1", "p7", "p9"],
  "n_classes": 3
}
```

## Flask Routes

```text
/modules/metric-learning-adapter/                    adapter debug page
/modules/metric-learning-adapter/health              module health
/modules/metric-learning-adapter/api/constraints     convert instruction to constraints
```

## Flask Debug Page Requirements

The page should show:

1. sample structured instruction input.
2. editable JSON textarea.
3. generated constraint payload.
4. validation errors.
5. note showing whether real metric-learning code is connected or mocked.

## Testing

Unit tests:

1. `same_class` becomes must-link constraints.
2. `different_class` becomes cannot-link constraints.
3. `split_into_n_classes` becomes split constraints.
4. `is_outlier` becomes outlier hint if supported.
5. `needs_clarification` is rejected.
6. `non_actionable` is rejected.

Flask route tests:

1. debug page returns 200.
2. constraints API returns payload for valid instruction.
3. constraints API returns error for invalid instruction.

Manual browser check:

1. open `/modules/metric-learning-adapter/`.
2. submit sample instruction.
3. confirm constraint JSON is visible.
4. submit invalid instruction and confirm clear error.

## Completion Criteria

This module is complete when instruction-to-constraint conversion is inspectable through Flask and isolated from chat UI.

