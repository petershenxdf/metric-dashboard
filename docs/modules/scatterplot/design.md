# Scatterplot Module Design

## Purpose

The scatterplot module is the main visual interface for projected points, clusters, outliers, and point selection.

It is not only a chart. It is the visual context that later chat instructions refer to.

## Responsibilities

1. Render projected points.
2. Color points by cluster.
3. Mark outliers.
4. Show hover or label information.
5. Support point selection.
6. Send selection actions to the selection module.
7. Provide a Flask debug page for visual inspection.

## Not Responsible For

1. Computing MDS.
2. Running clustering.
3. Running outlier detection.
4. Owning selection truth.
5. Parsing chat messages.
6. Running metric learning.

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
      "selected": true,
      "metadata": {}
    }
  ]
}
```

## Flask Routes

```text
/modules/scatterplot/                         scatterplot debug page
/modules/scatterplot/health                   module health
/modules/scatterplot/api/render-payload       points ready for rendering
/workflows/scatter-selection/                 scatterplot plus selection interaction
```

## Flask Debug Page Requirements

The page should show:

1. SVG or canvas scatterplot.
2. cluster colors.
3. outlier markers.
4. selected point styling.
5. current selected point IDs.
6. clear selection action.
7. JSON render payload preview.

The first version can use SVG and vanilla JavaScript.

## Interaction Rules

1. Clicking a point toggles selection.
2. Selection state is sent to the selection module.
3. Scatterplot reads current selection state before rendering.
4. Visual selected state should match selection module state.

## Testing

Unit tests:

1. render payload includes one item per projected point.
2. cluster and outlier fields are included.
3. selected state is applied from selection context.
4. unknown point IDs are rejected.

Flask route tests:

1. debug page returns 200.
2. render payload API returns points.
3. workflow page returns 200.

Manual browser check:

1. open `/modules/scatterplot/`.
2. confirm points are visible.
3. confirm cluster colors and outlier markers are visible.
4. click points and confirm selection state changes.
5. open `/workflows/scatter-selection/` to check interaction with selection module.

## Completion Criteria

This module is complete when the scatterplot can be visually inspected and point selection can be tested through Flask.

