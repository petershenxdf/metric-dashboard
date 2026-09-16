# State and API contracts

All endpoints return {"ok": true, "data": ..., "error": null, "diagnostics": {}} on success.

## Dataset and session

POST /api/datasets accepts JSON records or multipart CSV/JSON/MAT and returns DatasetVersion directly
under data. GET /api/datasets lists versions. Ground-truth roles cannot overlap visible/model roles.

POST /api/active-learning/sessions accepts dataset_version_id and optional config.
It returns data.session and data.state.
Config includes n_clusters=3, min_pts=3, batch_size=4, review_percentile_threshold=75,
optional label_budget, max_points=2000 and explanation-tree settings.
MinPts must be positive and smaller than the dataset size.

GET /api/active-learning/sessions/<session_id>/state returns current session, round, review,
legacy_round, dataset_version, analysis, rule_set, plot_points, active_labels and history.
focus_category optionally selects one of the six review IDs; it does not recompute anything.

## Frozen review snapshot

A round includes review instead of recommendation_plans.
review.categories contains six fixed definitions, pool_size, all_tied and recommended_point_ids.
review.rows contains point_id, model group, review_status, eligible, rereview_reason, duplicate_key,
and six scores keyed by category ID. Each score has raw, percentile, recommendable,
unavailable_reason, ineligibility_reason, comparison_point_ids and evidence.
Missing values are null, never zero placeholders.
The snapshot includes MinPts, thresholds, expansion_tree, instability settings/group members and
run errors. See [score definitions](modules/active_learning/design.md).

## Feedback

POST /api/active-learning/sessions/<session_id>/rounds/<round_id>/labels:

~~~json
{
  "expected_round_id": "alround_...",
  "expected_label_revision": 0,
  "category": "local_sparsity",
  "labels": [
    {"point_id": "p01", "label_dimension": "outlier_status", "label_value": false}
  ]
}
~~~

category defaults to model_instability. Valid label dimensions are semantic_class (nonempty human
type string), outlier_status (boolean), and uncertain. No plan_id is required.
The endpoint returns events, session and the next round. Stale round/revision returns 409;
invalid labels return 400. Events carry category, review version and resulting-round provenance.

Normal status now contributes a normal reference inside SSDBCODI without becoming a semantic
class label. Bootstrap excludes confirmed outliers before selecting candidates and can replace
rejected anchors. All-confirmed-outlier feedback is valid: it yields no normal groups, rScore=0,
and null e_max in detailed SSDBCODI point scores (no normal expansion path), never Infinity/NaN.
Existing round snapshots are not migrated or recalculated by reading them.

POST /api/active-learning/sessions/<session_id>/rounds/<round_id>/revert restores the chosen
historical branch without deleting label history. GET /api/active-learning/sessions/<session_id>/history
returns session, rounds and label_events.

## Explicit legacy upgrade

POST /api/active-learning/sessions/<session_id>/upgrade accepts expected_round_id.
Only a legacy round with no review snapshot can upgrade. It creates a new round without adding
labels or rewriting historical analysis. A stale head returns 409; an already upgraded round 400.

The old /categories/<category>/interpret endpoint has been removed (404).
DeepSeek, TranslationPacket, CategoryEvidenceCard and RecommendationPlanV2 are no longer contracts.
