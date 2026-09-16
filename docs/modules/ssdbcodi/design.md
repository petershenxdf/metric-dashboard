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

The implementation exposes rScore, lScore, simScore and tScore diagnostics. Full-path reachability
uses a deterministic Prim expansion tree on the mutual-reachability graph, with recorded edges
exported through AnalysisResult diagnostics. Tree paths preserve minimax barriers. This replaces
the previous direct-to-seed rScore calculation. The current feedback contract is versioned as
expansion_normal_feedback_v2 (historical expansion_minimax_v1 snapshots are unchanged).
The six-score review layer uses only human-supported references on those paths, not bootstrap
anchors. Raw review values and precise percentiles are available alongside plain-language reasons.

## Labels

Active semantic-class labels become stable human seeds through the active-learning service. True-outlier and normal labels constrain outlier interpretation. Uncertain labels remain historical evidence and are not seeds.

Bootstrap anchors provide a baseline. Explicit human labels take precedence for labeled points.

Normal feedback is a normal reachability reference, not a semantic class seed. Core rScore uses
the union of cluster seeds and explicit Normal references on the recorded tree. This can change
neighboring rScore/tScore values without inventing a class. Confirmed normals are excluded from
automatic outlier selection and receive the same weighted-distance group assignment as other
normal records. Human semantic seeds likewise supply normal support unless explicitly Outlier.

Bootstrap runs on eligible records after excluding confirmed outliers, including before density
candidate selection. It deterministically chooses replacement anchors if old anchors are rejected,
and reduces bootstrap K when fewer than K candidates remain. If every record is confirmed Outlier,
the result has no seeds or normal assignments: rScore is zero and e_max is null (no normal path).
No synthetic normal reference or infinite JSON value is introduced. Correcting one record back
to Normal permits bootstrap and normal assignment again. An empty automatic candidate set produces
no automatic outliers; it must never fall back to reclassifying protected normal references.

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
