# Projection Module Design

## Purpose

The projection module converts a validated feature matrix into 2D coordinates for visualization.

The initial method is MDS.

## Responsibilities

1. Receive `FeatureMatrix` from data workspace.
2. Compute 2D coordinates with MDS.
3. Preserve point IDs.
4. Return a `ProjectionResult`.
5. Provide a Flask page that visualizes the projection.
6. Provide API output for coordinates.

## Not Responsible For

1. Rendering final scatterplot interactions.
2. Coloring by cluster.
3. Marking outliers.
4. Managing selection.
5. Parsing chat messages.
6. Running metric learning.

## Target Files

```text
app/modules/projection/
  __init__.py
  schemas.py
  mds.py
  service.py
  fixtures.py
  routes.py
  templates/projection/index.html
  static/projection/projection.js

tests/modules/projection/
  test_service.py
  test_routes.py
```

## Current Status

Status: `working`

Step 2 implementation is complete enough for local inspection:

1. MDS service exists.
2. Projection schemas exist.
3. Flask debug page exists.
4. projection and state APIs exist.
5. `/workflows/data-projection/` exists.
6. unit tests and route tests exist.

## Input Contract

```json
{
  "point_ids": ["p1", "p2"],
  "feature_names": ["x", "y"],
  "values": [[0.1, 0.2], [0.8, 0.4]]
}
```

## Output Contract

```json
{
  "projection_id": "projection_mds_001",
  "method": "mds",
  "coordinates": [
    {
      "point_id": "p1",
      "x": 0.25,
      "y": -0.71
    }
  ]
}
```

## Public Service API

```python
project_feature_matrix(feature_matrix, projection_id=None)
```

## Flask Routes

```text
/modules/projection/                      projection debug page
/modules/projection/health                module health
/modules/projection/api/projection        projection JSON
/modules/projection/api/state             current module summary as JSON
/workflows/data-projection/               data table and projection together
```

## Flask Debug Page Requirements

The page should show:

1. an SVG scatterplot of projected fixture data.
2. a coordinate table.
3. method name.
4. projection ID.
5. JSON API link.
6. visual labels from fixture metadata when available.

For visual checking, use a classic dataset such as Iris and color by known label only inside the debug page. This debug color is not the production cluster color.

## Testing

Unit tests:

1. every input point gets exactly one coordinate.
2. output point IDs match input point IDs.
3. coordinates are finite.
4. repeated input gives stable output.
5. invalid matrices are rejected.

Flask route tests:

1. debug page returns 200.
2. API returns `projection_id`, `method`, and `coordinates`.
3. API coordinate count matches fixture point count.
4. state API returns module status and coordinate count.
5. data-projection workflow page returns 200.

Manual browser check:

1. run `python run.py`.
2. open `/modules/projection/`.
3. inspect the MDS SVG and coordinate table.
4. open `/workflows/data-projection/` to see data and projection together.

Validation commands:

```powershell
python -m compileall app tests run.py
python -m unittest discover -s tests
```

## Completion Criteria

The module is complete when projection behavior passes unit tests and can be visually inspected through Flask.
