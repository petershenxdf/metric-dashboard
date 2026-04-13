# Data Workspace Module Design

## Purpose

The data workspace module owns dataset creation, point identity, feature names, metadata, and feature matrix output.

Every later module should refer to points by stable `point_id`, never by raw row index.

## Responsibilities

1. Load or receive raw user data.
2. Normalize data into shared schemas.
3. Assign stable point IDs when missing.
4. Validate feature vectors.
5. Preserve point order.
6. Produce a `FeatureMatrix` for projection and algorithm adapters.
7. Provide a Flask debug page for inspecting dataset state.

## Not Responsible For

1. MDS projection.
2. Clustering.
3. Outlier detection.
4. Selection state.
5. Chat parsing.
6. Metric learning.

## Target Files

```text
app/modules/data_workspace/
  __init__.py
  schemas.py
  service.py
  fixtures.py
  routes.py
  templates/data_workspace/index.html
  static/data_workspace/data_workspace.js

tests/modules/data_workspace/
  test_service.py
  test_routes.py
```

## Current Status

Status: `working`

Step 1 implementation is complete enough for local inspection:

1. service logic exists.
2. Iris fixture exists.
3. Flask debug page exists.
4. dataset and matrix APIs exist.
5. unit tests and route tests exist.

## Core Schemas

Point:

```json
{
  "point_id": "p1",
  "features": [0.1, 0.8, 0.3],
  "metadata": {
    "label": "optional label"
  }
}
```

Dataset:

```json
{
  "dataset_id": "iris_sample",
  "points": [],
  "feature_names": ["sepal_length", "sepal_width"],
  "created_at": "timestamp"
}
```

Feature matrix:

```json
{
  "point_ids": ["p1", "p2"],
  "feature_names": ["x", "y"],
  "values": [[0.1, 0.2], [0.8, 0.4]]
}
```

## Public Service API

```python
create_dataset(raw_points, dataset_id=None, feature_names=None)
create_feature_matrix(dataset)
create_point_id_map(dataset)
```

## Flask Routes

```text
/modules/data-workspace/                 dataset debug page
/modules/data-workspace/health           module health
/modules/data-workspace/api/dataset      current fixture dataset as JSON
/modules/data-workspace/api/matrix       current feature matrix as JSON
/modules/data-workspace/api/state        current module summary as JSON
```

## Flask Debug Page Requirements

The page should show:

1. dataset ID.
2. feature names.
3. point table with ID, features, and metadata.
4. feature matrix preview.
5. JSON links for dataset and matrix.
6. fixture selector when multiple fixtures exist.

## Fixtures

Use small deterministic fixtures first:

1. tiny numeric dataset for unit tests.
2. classic Iris sample for browser inspection.

The fixture should make later projection and scatterplot pages visually meaningful.

## Testing

Unit tests:

1. valid input creates stable point IDs.
2. omitted point IDs become deterministic IDs.
3. empty input is rejected.
4. missing features are rejected.
5. duplicate point IDs are rejected.
6. feature matrix preserves point order.

Flask route tests:

1. debug page returns 200.
2. dataset API returns `dataset_id`, `points`, and `feature_names`.
3. matrix API returns `point_ids`, `feature_names`, and `values`.
4. state API returns point count, feature count, and matrix shape.

Manual browser check:

1. run `python run.py`.
2. open `/modules/data-workspace/`.
3. verify the point table and matrix match the fixture.

Validation commands:

```powershell
python -m compileall app tests run.py
python -m unittest discover -s tests
```

## Completion Criteria

The module is complete when both code tests and Flask-visible dataset inspection work.
