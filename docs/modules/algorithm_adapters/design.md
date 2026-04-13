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

## Current Status

Status: `working`

Step 3 implementation is complete enough for local inspection:

1. Local Outlier Factor runs first over the feature matrix.
2. Detected outliers are excluded from clustering.
3. deterministic KMeans runs on the remaining non-outlier points.
4. Flask debug page exposes adjustable `n_clusters`.
5. cluster, outlier, analysis, and state APIs exist.
6. `/workflows/default-analysis/` combines data, projection, outliers, and clusters.

The current algorithms are intentionally wrapped as replaceable adapters. The future
algorithm slot is reserved for integrated algorithms such as
[SSDBCODI](https://arxiv.org/abs/2208.05561), but SSDBCODI is not used in the
current implementation.

The service exposes an `AnalysisProvider` boundary. The current provider is
`SequentialLofThenKMeansProvider`; a future SSDBCODI provider should implement
the same boundary and return the same dashboard-facing result schemas.

The default Flask fixture is `default_analysis_outlier_debug`, not Iris. It uses
three compact two-dimensional clusters plus three distant outlier candidates so
the debug page reliably shows both clustering and outlier-detection behavior.

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
/modules/algorithm-adapters/api/analysis      combined outlier and cluster JSON
/modules/algorithm-adapters/api/state         module summary JSON
/workflows/default-analysis/                  projection plus default analysis
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

