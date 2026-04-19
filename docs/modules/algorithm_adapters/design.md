# Algorithm Adapters Module Design

## Purpose

The algorithm adapters module exposes the dashboard's clustering and outlier provider boundary.

The active provider is SSDBCODI. The legacy LOF-then-KMeans provider remains available for comparison, but downstream modules should call the adapter boundary rather than importing a concrete algorithm.

## Responsibilities

1. Convert dashboard feature or representation schemas into algorithm inputs.
2. Call the active analysis provider.
3. Keep legacy clustering and outlier functions available behind explicit provider selection.
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

1. `run_default_analysis()` uses `SsdbcodiProvider` by default.
2. SSDBCODI emits both cluster assignments and outlier flags in one provider run.
3. The debug page exposes adjustable bootstrap cluster count, MinPts, and contamination.
4. cluster, outlier, analysis, and state APIs exist.
5. `/workflows/default-analysis/` combines data, projection, outliers, and clusters.

The service exposes an `AnalysisProvider` boundary. The default provider is
`SsdbcodiProvider`, while `SequentialLofThenKMeansProvider` is retained as a
legacy explicit provider. Both return the same dashboard-facing result schemas.

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
/workflows/provider-feedback/                 adapter boundary plus standalone SSDBCODI diagnostics
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

