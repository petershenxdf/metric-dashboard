# Data Workspace Design

## Role

data_workspace defines the shared Dataset, PointRecord, and FeatureMatrix contracts used by projection and analysis modules. Its debug page uses a deterministic fixture.

Generic CSV, JSON, MAT import, preprocessing, artifact storage, and DatasetVersion ownership live in active_learning/data.py because those operations are session-versioned.

## Contracts

Dataset preserves stable point IDs, source feature values, metadata, and optional display labels.

FeatureMatrix contains ordered point IDs, model feature names, and finite numeric rows. Its row order must match point_ids exactly.

The active-learning PreparedDataset extends these contracts with raw records, transformed matrix artifacts, schema roles, fingerprints, and a transformation map.

## Invariants

- point IDs are unique and non-empty;
- feature rows are rectangular and finite;
- source metadata is not silently treated as a model feature;
- ground truth is never included in the model matrix;
- modules consume schemas/services rather than importing workflow code.

## Debug Surface

~~~text
/modules/data-workspace/
/modules/data-workspace/health
/modules/data-workspace/api/state
~~~

The lab validates base contracts only. Product dataset import is available from the active-learning dashboard and /api/datasets.

## Tests

Tests cover schema validation, deterministic fixture conversion, route envelopes, and feature-matrix ordering.
