# Projection Design

## Role

projection converts a FeatureMatrix into deterministic two-dimensional MDS coordinates for display.

Projection supports visual inspection and selection. It does not determine clusters, outliers, recommendations, or labels.

## Input And Output

Input: ordered finite FeatureMatrix.

Output: ProjectionResult containing method, projection ID, and one x/y coordinate per input point ID.

The active-learning round snapshots the projection result so a restored round renders the same view.

## Invariants

- coordinate point IDs and order match the matrix;
- identical matrix/config input produces identical output;
- projection coordinates never enter decision-tree rule generation;
- recommendation evidence uses raw/model features and analysis state, not visual distance alone;
- the scatterplot may link guidance to a point without changing projection state.

## Debug Surface

~~~text
/modules/projection/
/modules/projection/health
/modules/projection/api/projection
/modules/projection/api/state
~~~

The fixture coloring on the lab page is diagnostic only.

## Tests

Tests cover dimensions, stable identity, point ordering, finite coordinates, route envelopes, and isolated module rendering.
