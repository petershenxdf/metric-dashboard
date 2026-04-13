# Selection Module Design

## Purpose

The selection module owns selected and unselected point state.

It grounds user phrases such as "these points", "selected points", and "unselected points".

It does not decide what the selected points mean. That semantic layer belongs to the labeling module or the intent instruction module.

## Responsibilities

1. Store selected point IDs.
2. Add points to selection.
3. Remove points from selection.
4. Replace selection.
5. Clear selection.
6. Return selected and unselected point IDs.
7. Expose selection state through Flask.

## Not Responsible For

1. Rendering scatterplot geometry.
2. Parsing chat messages.
3. Running clustering or outlier detection.
4. Running metric learning.
5. Assigning selected points to clusters or outlier labels.

## Target Files

```text
app/modules/selection/
  __init__.py
  schemas.py
  store.py
  service.py
  fixtures.py
  routes.py
  templates/selection/index.html
  static/selection/selection.js

tests/modules/selection/
  test_service.py
  test_routes.py
```

## State Contract

```json
{
  "dataset_id": "iris_sample",
  "selected_point_ids": ["p1", "p7"],
  "unselected_point_ids": ["p2", "p3"],
  "selected_count": 2,
  "unselected_count": 2
}
```

## Public Service API

```python
select_points(point_ids)
deselect_points(point_ids)
replace_selection(point_ids)
clear_selection()
get_selection_context(dataset)
```

Downstream modules should consume this context instead of reading selection internals directly.

```json
{
  "source": "selection",
  "dataset_id": "iris_sample",
  "selected_point_ids": ["p1", "p7"],
  "unselected_point_ids": ["p2", "p3"]
}
```

## Flask Routes

```text
/modules/selection/                       selection debug page
/modules/selection/health                 module health
/modules/selection/api/state              selection state JSON
/modules/selection/api/select             select points
/modules/selection/api/deselect           deselect points
/modules/selection/api/replace            replace selection
/modules/selection/api/clear              clear selection
```

## Flask Debug Page Requirements

The page should show:

1. clickable fixture point list.
2. selected point IDs.
3. unselected point IDs.
4. JSON preview of selection context.
5. clear button.
6. link to the labeling workflow when available.

## Testing

Unit tests:

1. select known points.
2. reject unknown point IDs.
3. deselect points.
4. clear selection.
5. return selected and unselected IDs.

Flask route tests:

1. debug page returns 200.
2. state API returns selected and unselected IDs.
3. select and clear APIs update state.

Manual browser check:

1. open `/modules/selection/`.
2. click several points.
3. confirm visible JSON updates.
4. clear selection and confirm state resets.

## Completion Criteria

This module is complete when selection state can be tested in code and manipulated visibly through Flask.

The first downstream integration should be `/workflows/selection-labeling/`, where selected points are converted into manual annotations.
