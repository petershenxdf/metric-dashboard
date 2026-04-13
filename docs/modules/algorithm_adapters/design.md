# Algorithm Adapters Module Design

## Purpose

The algorithm adapters module wraps existing clustering and outlier detection algorithms.

The existing algorithms are treated as fixed external dependencies. This module adapts their input and output to dashboard schemas.

## Responsibilities

1. Convert dashboard feature or representation schemas into algorithm inputs.
2. Call existing clustering logic.
3. Call existing outlier detection logic.
4. Convert algorithm outputs back to point-ID-based schemas.
5. Surface diagnostics in Flask.
6. Make mock adapter output visible before real algorithms are connected.

## Not Responsible For

1. Redesigning clustering.
2. Redesigning outlier detection.
3. Running metric learning.
4. Rendering scatterplot UI.
5. Parsing chat messages.

## Target Files

```text
app/modules/algorithm_adapters/
  __init__.py
  schemas.py
  clustering.py
  outliers.py
  service.py
  fixtures.py
  routes.py
  templates/algorithm_adapters/index.html

tests/modules/algorithm_adapters/
  test_clustering.py
  test_outliers.py
  test_routes.py
```

## Output Contracts

Cluster assignment:

```json
{
  "cluster_run_id": "cluster_001",
  "assignments": [
    {
      "point_id": "p1",
      "cluster_id": "c1"
    }
  ]
}
```

Outlier result:

```json
{
  "outlier_run_id": "outlier_001",
  "scores": [
    {
      "point_id": "p1",
      "score": 0.03,
      "is_outlier": false
    }
  ]
}
```

## Flask Routes

```text
/modules/algorithm-adapters/                  adapter debug page
/modules/algorithm-adapters/health            module health
/modules/algorithm-adapters/api/clusters      cluster assignments JSON
/modules/algorithm-adapters/api/outliers      outlier scores JSON
```

## Flask Debug Page Requirements

The page should show:

1. selected fixture dataset or projection.
2. cluster assignments table.
3. outlier score table.
4. diagnostics showing real vs mock algorithm mode.
5. algorithm name/config if available.

## Testing

Unit tests:

1. adapter preserves point IDs.
2. adapter calls expected algorithm boundary.
3. invalid algorithm output is rejected.
4. output lengths match input point count.

Flask route tests:

1. debug page returns 200.
2. cluster API returns assignment list.
3. outlier API returns score list.

Manual browser check:

1. open `/modules/algorithm-adapters/`.
2. verify clusters and outlier scores are visible.
3. verify the page clearly shows whether outputs are mock or real.

## Completion Criteria

This module is complete when existing or mock algorithm outputs can be inspected through Flask without touching algorithm internals.

