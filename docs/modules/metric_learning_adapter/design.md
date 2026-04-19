# Metric-Learning Adapter Module Design (Path A)

## Purpose

The metric-learning adapter converts structured instructions and manual labels into a learned Mahalanobis distance metric.

It is the single boundary between dashboard-facing feedback and the actual metric learning library **on Path A**. Path B has its own sibling module, `direct_feedback_adapter`, which compiles the same inputs into SSDBCODI-native inputs instead of a metric. Keeping the two adapters separate makes each path independently testable and keeps their error codes, diagnostics, and debug pages from being entangled.

It should not receive raw chat text. It should only receive validated structured instructions and labeling annotations.

## Relationship to Path B

Both `metric_learning_adapter` (this module) and `direct_feedback_adapter` consume the **same** `StructuredInstruction` and labeling annotation snapshot. The difference is in what they produce:

| Adapter | Output | Consumed by |
|---|---|---|
| `metric_learning_adapter` (Path A) | `ConstraintSet` + `LearnedMetric` (M, L) | `metric_refinement_orchestrator` |
| `direct_feedback_adapter` (Path B) | `DirectFeedbackPlan` (seeds, feature_scale, param_overrides) | `direct_refinement_orchestrator` |

This adapter keeps `split_cluster` and `reclassify_outlier` **deferred** because metric change alone cannot drive them. Those intents are first-class on Path B. See `docs/modules/direct_feedback_adapter/design.md`.

## Responsibilities

1. Accept a `ConstraintSet` built from `StructuredInstruction` (chat) and labeling annotations.
2. Validate instruction source, status, and referenced point IDs.
3. Reject incomplete and non-actionable instructions.
4. Run a replaceable metric learner and return a Mahalanobis matrix `M` (plus its Cholesky factor `L`).
5. Apply `L` as a linear pre-transform to the feature matrix so projection and algorithm adapters can consume it unchanged.
6. Provide a Flask page for inspecting the constraint set, learned `M`, and the transformed feature space.

## Not Responsible For

1. Parsing natural language.
2. Rendering chat UI.
3. Managing selection state.
4. Running clustering directly.
5. Running outlier detection directly.
6. Owning manual label state.

## Target Files

```text
app/modules/metric_learning_adapter/
  __init__.py
  schemas.py
  constraint_builder.py
  providers/
    base.py
    identity.py
    itml.py
  adapter.py
  fixtures.py
  routes.py
  templates/metric_learning_adapter/index.html

tests/modules/metric_learning_adapter/
  test_constraint_builder.py
  test_providers.py
  test_adapter.py
  test_routes.py
```

## Pipeline

```text
labeling annotations        ──┐
StructuredInstruction       ──┼─► constraint_builder ─► ConstraintSet ─► MetricLearnerProvider ─► M
selection groups (read-only)──┘                                                                  │
                                                                                                 ▼
                                                    apply L = chol(M) as pre-transform: X' = X · L
                                                                                                 │
                                                                                                 ▼
                                                    projection and algorithm_adapters run on X'
```

Projection and algorithm adapters are not changed. They receive a transformed feature matrix rather than the raw one.

## Constraint Builder

`constraint_builder.py` is a pure function module. It merges the three input sources into a unified `ConstraintSet`:

1. Labeling `assign_cluster` annotations - intra-label similar pairs.
2. Labeling `mark_outlier` annotations - outlier handling is owned by labeling directly; the adapter does not translate these into metric pairs in Phase 1.
3. Structured instruction constraints, by intent type (see mapping below).

### Intent to Constraint Mapping (Phase 1)

| Intent | Produces |
|--------|----------|
| `feature_weight` | Entry in `feature_scale` dict (see below). Not a pair constraint. |
| `group_similar` | Sampled must-link pairs between the two groups. |
| `group_dissimilar` | Sampled cannot-link pairs between the two groups. |
| `merge_clusters` | Sampled must-link pairs across all member clusters. |
| `anchor_point` | Must-link pairs from anchor to each target group point. |
| `ignore_cluster` | No pairs generated from that cluster this round. |

### Deferred Intents on Path A

On this path the following intents cannot be realized through a distance metric and are deferred:

1. `split_cluster` - would require intra-cluster cannot-link pairs plus a `k` increment in the clustering provider. The metric change alone cannot force KMeans to split a cluster, so a learned `M` cannot produce a split here.
2. `reclassify_outlier` - SSDBCODI's automatic outlier decision still depends on score ranking and contamination. A metric change may not move a point across the contamination threshold.

The constraint builder rejects these intents with error code `intent_deferred` and a `suggested_strategy: "direct_ssdbcodi"` hint. The user can route them through Path B (`direct_feedback_adapter` + `direct_refinement_orchestrator`) where they are first-class inputs rather than deferred errors.

### Pair Sampling

For any intent that generates pairs between groups of size `n` and `m`, the builder samples `min(n*m, max_pairs_per_intent)` pairs rather than generating all of them. Default `max_pairs_per_intent = 50`. This prevents constraint counts from blowing up ITML solve time on large groups.

### Conflict Detection

The builder performs a simple conflict check before handing off to the learner:

1. The same pair appearing as both must-link and cannot-link is flagged.
2. When conflicts are detected, the adapter returns `ok: false` with a `conflicts` field so the chatbox can ask the user to resolve them.

## ConstraintSet Schema

```json
{
  "must_link": [["p1", "p7"], ["p1", "p9"]],
  "cannot_link": [["p1", "p23"]],
  "feature_scale": {
    "petal_length": 2.0,
    "sepal_width": 0.5
  },
  "excluded_clusters": ["cluster_5"],
  "conflicts": [],
  "diagnostics": {
    "pair_count": 12,
    "sampled_from": 350,
    "sources": ["labeling", "chat_intent"]
  }
}
```

`feature_scale` maps feature name to multiplier. `1.0` means unchanged. Values above 1 emphasize the feature, below 1 de-emphasize it, and `0.0` drops it.

## Metric Learner Provider Protocol

```python
class MetricLearnerProvider(Protocol):
    name: str
    def fit(self, X: np.ndarray, constraints: ConstraintSet) -> LearnedMetric: ...
```

`LearnedMetric` contains:

```python
@dataclass
class LearnedMetric:
    M: np.ndarray         # Mahalanobis matrix
    L: np.ndarray         # Cholesky factor, used as linear transform
    provider: str
    n_constraints_used: int
    diagnostics: dict
```

Built-in providers:

1. `IdentityProvider` - returns `M = I`. Used when no constraints exist or for cold start. Keeps Step 1-6 workflows functional.
2. `ItmlProvider` - default learner. Wraps `metric-learn`'s ITML. Handles `must_link`, `cannot_link`, and accepts `feature_scale` via pre-scaling of `X` before the fit.

Future providers (documented but not implemented in Phase 1):

3. `LmnnProvider` - for cases where per-point class labels are strong.
4. `MmcProvider` - stricter variant of pair-based learning.

Provider is chosen from `settings.json` or a request parameter on the Flask API for experimentation.

## Feature Scaling Path

`feature_weight` intents are applied before the metric learner sees the data:

```text
S         = diag(feature_scale)
X_scaled  = X · S
M_scaled  = learner.fit(X_scaled, pair_constraints)
L_total   = S · chol(M_scaled)
X'        = X · L_total
```

Row vectors are transformed on the right, so the composite transform is `S` applied first, then `chol(M_scaled)`. This separates feature importance from pair-based similarity learning. It also means that a user who only provides `feature_weight` hints still gets a meaningful metric change, which ITML alone would not produce.

## Flask Routes

```text
/modules/metric-learning-adapter/                       adapter debug page
/modules/metric-learning-adapter/health                 module health
/modules/metric-learning-adapter/api/constraints        build ConstraintSet from inputs
/modules/metric-learning-adapter/api/fit                run provider, return learned metric
/modules/metric-learning-adapter/api/providers          list available providers
```

## Flask Debug Page Requirements

The page should show:

1. sample structured instruction input and labeling annotation input.
2. editable JSON textareas for both.
3. current active provider with a dropdown to switch.
4. generated `ConstraintSet` with pair count and conflict list.
5. learned `M` preview (heatmap or numeric grid for small feature counts).
6. transformed feature matrix preview.
7. note showing whether the provider is real or mocked.
8. explicit note that `split_cluster` and `reclassify_outlier` intents are deferred on this path, with a pointer to Path B (`direct_feedback_adapter`).

## Testing

Unit tests (constraint_builder):

1. `group_similar` produces must-link pairs, count bounded by `max_pairs_per_intent`.
2. `group_dissimilar` produces cannot-link pairs.
3. `merge_clusters` with two clusters produces cross-cluster must-link pairs.
4. `feature_weight` populates `feature_scale`, not pair lists.
5. `anchor_point` produces must-link pairs from anchor to every target.
6. `ignore_cluster` excludes that cluster's points from pair generation.
7. `split_cluster` is rejected with `intent_deferred`.
8. `reclassify_outlier` is rejected with `intent_deferred`.
9. Conflicting must-link/cannot-link pairs are detected and reported.
10. Labeling `assign_cluster` annotations become intra-label must-link pairs.

Unit tests (providers):

1. `IdentityProvider.fit` returns `M = I` regardless of constraints.
2. `ItmlProvider.fit` with no constraints returns something close to identity.
3. `ItmlProvider.fit` with similar pairs reduces distance between those pairs in the learned metric.
4. `ItmlProvider.fit` with `feature_scale` produces an `L` that reflects the scaling.
5. Provider factory returns the configured provider.

Unit tests (adapter):

1. Full pipeline from structured instruction + labels to `LearnedMetric` works on fixture data.
2. Empty input returns `IdentityProvider` result and no errors.
3. Deferred intents return clear error.

Flask route tests:

1. debug page returns 200.
2. constraints API returns `ConstraintSet` for valid input.
3. constraints API returns error for invalid input and for deferred intents.
4. fit API returns `LearnedMetric` for valid constraints.
5. providers API lists `identity` and `itml`.

Manual browser check:

1. open `/modules/metric-learning-adapter/`.
2. submit a sample instruction with `group_similar`.
3. confirm constraint JSON is visible with pair count.
4. click fit and confirm `M` preview updates.
5. submit a `split_cluster` instruction and confirm the error clearly says the intent is deferred.

## Completion Criteria

This module is complete when:

1. Constraint building is testable on both chat-derived and labeling-derived input.
2. `IdentityProvider` and `ItmlProvider` both run through Flask.
3. The learned metric applies cleanly as a pre-transform to the feature matrix consumed by projection and algorithm adapters.
4. Deferred intents are explicitly rejected with a documented error code.
5. The debug page makes the constraint set and learned metric inspectable in the browser.
