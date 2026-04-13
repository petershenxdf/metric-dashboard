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
8. Save reusable named selections as selection groups.
9. Re-select a saved group by group name or group ID.

## Not Responsible For

1. Rendering scatterplot geometry.
2. Parsing chat messages.
3. Running clustering or outlier detection.
4. Running metric learning.
5. Assigning selected points to clusters or outlier labels.

Selection groups are not semantic labels. They are named point sets that make
it easy to return to a previous selection. The labeling module is still
responsible for deciding whether selected points mean "same class", "outlier",
or another annotation type.

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

## Current Status

Status: `working`

Step 4 implementation is complete enough for local inspection:

1. selection state is stored in local in-memory debug state.
2. `select`, `deselect`, `replace`, `toggle`, and `clear` actions are supported.
3. action payloads include `source` and `mode` fields so future selection types can be added without changing downstream contracts.
4. supported sources include `api`, `point_click`, `lasso`, `rectangle`, `manual_list`, `workflow_fixture`, and `selection_group`.
5. `/modules/selection/` exposes a clickable point list, action lab, state preview, and context preview.
6. `/workflows/selection-context/` shows Data Workspace point IDs converted into reusable selection context.
7. users can save the current selection as a named selection group, click that group later to replace the active selection, and delete saved groups.
8. `/workflows/analysis-selection/` overlays selection onto the Step 1-3 data/projection/analysis output for combined testing.
9. the combined workflow supports a dataset dropdown, click selection, and rectangle selection without exposing mode choices to the user.

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

## Selection Group Contract

```json
{
  "group_id": "group_001",
  "group_name": "interesting pair",
  "dataset_id": "iris_sample",
  "point_ids": ["p1", "p7"],
  "point_count": 2,
  "metadata": {}
}
```

Selection group names are unique within the current in-memory selection store.
Selecting a group applies a normal `replace` selection action with
`source: "selection_group"`.

## Public Service API

```python
select_points(point_ids)
deselect_points(point_ids)
replace_selection(point_ids)
toggle_points(point_ids)
clear_selection()
get_selection_context(dataset)
apply_selection_action(action)
save_selection_group(group_name, point_ids=None)
list_selection_groups()
select_selection_group(group_id)
delete_selection_group(group_id)
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
/modules/selection/api/toggle             toggle selection membership
/modules/selection/api/clear              clear selection
/modules/selection/api/context            selection context JSON
/modules/selection/api/reset              reset local debug fixture
/modules/selection/api/groups             list or create saved selection groups
/modules/selection/api/groups/<id>/select replace selection with a saved group
/modules/selection/api/groups/<id>        delete a saved group
/workflows/selection-context/             data workspace plus selection context
/workflows/analysis-selection/            data, projection, analysis, and selection visual test
```

Action payload:

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

The current action endpoints infer the action from the route, for example
`POST /modules/selection/api/replace`. `clear` ignores `point_ids`.

Save current selection as a group:

```json
{
  "group_name": "interesting pair"
}
```

Save an explicit point set as a group:

```json
{
  "group_name": "setosa examples",
  "point_ids": ["setosa_001", "setosa_002"]
}
```

## Flask Debug Page Requirements

The page should show:

1. clickable fixture point list.
2. selected point IDs.
3. unselected point IDs.
4. JSON preview of selection context.
5. clear button.
6. saved selection group form.
7. saved selection group list with select and delete actions.
8. link to the labeling workflow when available.

## Testing

Unit tests:

1. select known points.
2. reject unknown point IDs.
3. deselect points.
4. clear selection.
5. return selected and unselected IDs.
6. replace selection.
7. toggle selection.
8. preserve action source and metadata for future UI gestures.
9. save a named selection group from current selection.
10. select and delete saved groups.

Flask route tests:

1. debug page returns 200.
2. state API returns selected and unselected IDs.
3. select and clear APIs update state.
4. all selection action APIs return the standard JSON envelope.
5. selection-context workflow returns 200.
6. group APIs can save current selection, select a group, delete a group, and reject duplicate names.
7. analysis-selection workflow returns the combined Step 1-4 page and state API.
8. analysis-selection workflow supports dataset switching plus click and rectangle selection.

Manual browser check:

1. open `/modules/selection/`.
2. click several points.
3. confirm visible JSON updates.
4. clear selection and confirm state resets.
5. try manual action lab with `select`, `deselect`, `replace`, `toggle`, and `clear`.
6. save the current selection as a named group.
7. change the active selection, then click the group name and confirm the saved points are selected again.
8. delete the group and confirm it disappears.
9. open `/workflows/selection-context/` and confirm selection context is visible.
10. open `/workflows/analysis-selection/` and confirm projected point clicks update selection on the combined plot.
11. drag a rectangle and confirm points inside the region are added to the active selection.

## Completion Criteria

This module is complete when selection state can be tested in code and manipulated visibly through Flask.

The first downstream integration should be `/workflows/selection-labeling/`, where selected points are converted into manual annotations.
