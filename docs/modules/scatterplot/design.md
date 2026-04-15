# Scatterplot Module Design

## Purpose

The scatterplot module is the main visual interface for projected points, clusters, outliers, and point selection.

It is not only a chart. It is the visual context that later chat instructions refer to.

It can expose direct label controls near the plot, but label state is owned by the labeling module.

## Responsibilities

1. Render projected points.
2. Color points by cluster.
3. Mark outliers.
4. Show hover or label information.
5. Support point selection.
6. Send selection actions to the selection module.
7. Render existing manual labels when labeling state is provided.
8. Send explicit label actions to the labeling module when the workflow includes it.
9. Provide a Flask debug page for visual inspection.

## Not Responsible For

1. Computing MDS.
2. Running clustering.
3. Running outlier detection.
4. Owning selection truth.
5. Owning label or annotation truth.
6. Parsing chat messages.
7. Running metric learning.

## Target Files

```text
app/modules/scatterplot/
  __init__.py
  schemas.py
  service.py
  fixtures.py
  routes.py
  templates/scatterplot/index.html
  static/scatterplot/scatterplot.js
  static/scatterplot/scatterplot.css

tests/modules/scatterplot/
  test_service.py
  test_routes.py
```

## Render Payload

```json
{
  "points": [
    {
      "point_id": "p1",
      "x": 0.2,
      "y": -0.7,
      "cluster_id": "c1",
      "is_outlier": false,
      "manual_label": null,
      "selected": true,
      "metadata": {}
    }
  ]
}
```

## Current Status

Status: `working`

Step 6 implementation is complete enough for local inspection:

1. scatterplot service builds a stable render payload from projection, cluster,
   outlier, selection, and label state.
2. `/modules/scatterplot/` shows projected points with cluster colors, outlier
   markers, selected point indicators, and render JSON.
3. `/modules/scatterplot/api/render-payload` exposes points ready for rendering.
4. `/workflows/scatter-selection/` verifies scatterplot point clicks update
   selection through the selection module.
5. `/workflows/scatter-labeling/` verifies selected points can be labeled
   through the labeling module while scatterplot only renders effective state.
6. The Step 1-6 pages preserve Step 4 selection behavior: click selection,
   rectangle selection, saved selection groups, and adjustable `n_clusters`.

## Flask Routes

```text
/modules/scatterplot/                         scatterplot debug page
/modules/scatterplot/health                   module health
/modules/scatterplot/api/render-payload       points ready for rendering
/workflows/scatter-selection/                 scatterplot plus selection interaction
/workflows/scatter-labeling/                  scatterplot plus selection and labeling interaction
```

## Flask Debug Page Requirements

The page should show:

1. SVG or canvas scatterplot.
2. cluster colors.
3. outlier markers.
4. selected point styling.
5. current selected point IDs.
6. visible manual labels when present.
7. clear selection action.
8. optional label controls when labeling is connected.
9. JSON render payload preview.

The first version can use SVG and vanilla JavaScript.

## Interaction Rules

1. Clicking a point toggles selection.
2. Selection state is sent to the selection module.
3. Scatterplot reads current selection state before rendering.
4. Visual selected state should match selection module state.
5. Label actions are sent to the labeling module.
6. Visual label state should match labeling module state.

## Testing

Unit tests:

1. render payload includes one item per projected point.
2. cluster and outlier fields are included.
3. selected state is applied from selection context.
4. manual label state is included when provided.
5. unknown point IDs are rejected.

Flask route tests:

1. debug page returns 200.
2. render payload API returns points.
3. workflow page returns 200.

Manual browser check:

1. open `/modules/scatterplot/`.
2. confirm points are visible.
3. confirm cluster colors and outlier markers are visible.
4. click points and confirm selection state changes.
5. drag a rectangle and confirm points inside it are selected.
6. save a selected set as a named group, change selection, then restore the group.
7. change `n_clusters` and confirm cluster labels and label options update.
8. open `/workflows/scatter-selection/` to check interaction with selection module.
9. open `/workflows/scatter-labeling/` and confirm selected points can be annotated.

## Completion Criteria

This module is complete when the scatterplot can be visually inspected and point selection can be tested through Flask.
