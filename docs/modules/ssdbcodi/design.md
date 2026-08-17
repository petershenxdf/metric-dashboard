# SSDBCODI Design

## Role

SSDBCODI is the active source of truth for semi-supervised clustering and integrated outlier detection.

It consumes a finite FeatureMatrix plus current effective human labels and returns cluster assignments, outlier results, and auditable per-point scores. It does not use the decision tree or DeepSeek.

## Analysis Flow

~~~text
FeatureMatrix
  -> deterministic density-safe bootstrap groups
  -> centroid-nearest normal anchors
  -> apply current valid human seeds/constraints
  -> compute SSDBCODI neighborhood and similarity evidence
  -> integrated cluster and outlier result
~~~

The implementation persists rScore, lScore, simScore, and tScore diagnostics. These values may support deterministic recommendation ranking and technical audit but should be translated into qualitative facts before appearing in primary user guidance.

## Labels

Active semantic-class labels become stable human seeds through the active-learning service. True-outlier and normal labels constrain outlier interpretation. Uncertain labels remain historical evidence and are not seeds.

Bootstrap anchors provide a baseline. Explicit human labels take precedence for labeled points.

## Outputs

The provider returns shared AnalysisResult, ClusterResult, and OutlierResult schemas through algorithm_adapters. Downstream code relies on these contracts rather than importing algorithm internals.

## Invariants

- clustering and outlier detection are integrated in this provider;
- identical data, parameters, and effective labels produce identical output;
- all scores are finite;
- manual label provenance remains outside the algorithm;
- tree rules never replace analysis assignments;
- ground truth never enters the provider;
- oversized inputs fail through the declared capability limit.

## Debug Surface

~~~text
/modules/ssdbcodi/
/modules/ssdbcodi/health
/modules/ssdbcodi/api/state
/modules/ssdbcodi/api/run
/modules/ssdbcodi/api/label
~~~

The module lab offers deterministic synthetic fixtures for algorithm inspection. The Wine dataset is tested through the generic active-learning path.

## Tests

Tests cover score formulas and ranges, deterministic execution, bootstrap behavior, manual labels, cluster/outlier contracts, store history, fixture isolation, and routes.

The bundled ssdbcodi algorithm.pdf remains the algorithm reference document.
