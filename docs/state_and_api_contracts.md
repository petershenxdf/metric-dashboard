# State And API Contracts

## Ownership

| State | Owner |
| --- | --- |
| Imported schema, raw artifacts, model matrix | active_learning data layer |
| Session, round, label, history | active_learning service and store |
| Projection coordinates | projection service, snapshotted per round |
| Cluster and outlier analysis | SSDBCODI through algorithm_adapters |
| Explanation rules | rule_panel |
| Candidate ranking and point selection | deterministic recommendation engine |
| Category evidence facts and statuses | active_learning evidence service |
| Human-readable explanation | active_learning translation |
| Transient module-lab selection | selection |
| Visual rendering | scatterplot and final workflow template |

## DatasetVersion

DatasetVersion is immutable and contains dataset_version_id, dataset_id, source format, entity name, content/version fingerprints, schema roles, preprocessing version/config, feature transformation map, artifact references, and row/feature counts.

Ground-truth columns may be stored in isolated raw artifacts for evaluation. They must not appear in FeatureMatrix, plot payloads, recommendation evidence, or TranslationPacket.

## ActiveLearningSession

A session binds one DatasetVersion to SessionConfig, label vocabulary, label budget, current round, current label revision, and lifecycle status.

SessionConfig includes analysis and tree parameters, candidate/batch sizes, preprocessing assumptions, and provider capability limits.

## ActiveLearningRound

A round is an immutable analysis snapshot with:

- round_id, index, and optional parent;
- label revision;
- analysis, projection, RuleSet, and display RuleSet;
- RecommendationPlanV2 values for all categories;
- RoundDelta from the parent;
- lifecycle status and stop suggestion;
- timestamps and diagnostics.

Round 0 is the unlabeled baseline. A valid label commit marks the current round labels_committed and creates the next round.

## LabelEvent

Label dimensions are semantic_class, outlier_status, and uncertain.

An event records stable point ID, dimension, value, source round, resulting
child round, plan/category provenance, recommendation membership, supersedes
relation, status, and timestamp. A correction supersedes the prior active
event rather than deleting history. The resulting round disambiguates branches
created from the same source round.

Uncertain events are recorded but do not become SSDBCODI seeds.

## RecommendationPlanV2

The canonical plan includes:

- plan/session/round/dataset/preprocessing/label revisions;
- focus category and target rules;
- complete candidate_pool_point_ids and candidate_rankings;
- ordered recommended_point_ids and highlighted_point_ids;
- candidate and canonical recommended point profiles;
- excluded/deferred points and recheck reasons;
- score components, selection reasons, and cross-category coverage;
- history context and previous-plan diff;
- category explanation, evidence-policy version, and ordered evidence cards;
- immutable fields, stop reason, and diagnostics.

Recommended and highlighted IDs must preserve order and be subsets of the candidate pool. The LLM cannot change this contract.

`history_context.recommendation_history` stores separate computed, shown,
selected, and labeled counts. Only shown history contributes a repetition
penalty.

## CategoryEvidenceCard

`CategoryEvidenceCard` is the canonical explanation evidence for one
recommended point. It contains:

- `point_id`, category, delegated evidence category, and policy version;
- all category-specific `evidence_bullets` in fixed order;
- comparison targets and fields to compare;
- round context.

Each `EvidenceBullet` contains a stable dimension ID and question, one of
`yes`, `partly`, `no`, or `insufficient`, deterministic headline, plain fact,
point connection, labeling value, evidence fact IDs, and folded technical
details. `point_connection` states why the finding applies to this record;
`labeling_value` states exactly what a human label would support, challenge, or
make possible in a later round.

Facts come from deterministic analysis in the complete model feature space.
The 2D projection may only be used for an agreement check. Ground truth is
never evidence.

## TranslationPacket And PointGuidance

TranslationPacket is a whitelist-filtered view of one plan. It includes only
relevant rules, recommended profiles, CategoryEvidenceCards without technical
details, allowed labels, prior label context, and round change facts.

PointGuidance provides one entry per recommended point:

- category explanation;
- ordered evidence bullets with immutable questions, statuses, and fact IDs;
- one direct answer, one supporting observation, and one `why_this_point`
  statement for every evidence dimension;
- exact comparison target IDs;
- how-to-label guidance;
- at most two conditional outcomes.

Provider output must preserve plan ID, category, target rules, recommended
IDs/order, dimension IDs/order, dimension statuses, fact IDs, and comparison
target IDs. Invalid prose for one bullet is replaced only for that bullet.
Changed immutable fields or provider failure uses the complete deterministic
fallback.

## APIs

### Datasets

~~~text
GET  /api/datasets
POST /api/datasets
~~~

POST accepts multipart CSV/JSON/MAT input or a JSON records payload. It returns DatasetVersion metadata, never the complete raw dataset.

### Sessions

~~~text
POST /api/active-learning/sessions
GET  /api/active-learning/sessions/<session_id>/state
GET  /api/active-learning/sessions/<session_id>/history
~~~

Session creation computes Round 0. State returns the current round, selected
category plan and evidence cards, deterministic guidance, plot payload,
labels, and diagnostics.

### Labels

~~~text
POST /api/active-learning/sessions/<session_id>/rounds/<round_id>/labels
~~~

Required concurrency fields are expected_round_id, expected_label_revision, and plan_id. A stale submission returns HTTP 409 with error code stale_round.

Each label item identifies point_id, dimension, and value. Labels may target recommended points or explicit user selection, but provenance records the difference.

### Revert

~~~text
POST /api/active-learning/sessions/<session_id>/rounds/<round_id>/revert
~~~

Revert retracts later effective events and restores the selected round as the session head without deleting history.

Effective events are reconstructed from parent ancestry and each event's
resulting round, never from round index alone.

### Explain

~~~text
POST /api/active-learning/sessions/<session_id>/rounds/<round_id>/categories/<category>/interpret
~~~

provider_kind may be deepseek or mock. A category without a typical case skips
the provider. Only this POST endpoint may make a provider call. Diagnostics
disclose requested/returned model, usage, finish reason, system fingerprint,
prompt/evidence versions, partial fallback fields, cache status, and fallback
reason.

## Error Semantics

- 400: invalid dataset, session config, labels, round action, or interpretation request.
- 404: unknown session or resource.
- 409: stale round/revision/plan submission.
- Provider errors do not invalidate the round; they return fallback guidance with explicit diagnostics.
