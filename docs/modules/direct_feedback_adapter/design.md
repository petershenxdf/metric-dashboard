# Direct Feedback Adapter Module Design (Path B)

## Purpose

The direct feedback adapter is the Path B counterpart to `metric_learning_adapter`.
It converts the same structured user feedback (labeling annotations plus the
current `StructuredInstruction`) into **SSDBCODI-native inputs** instead of a
Mahalanobis metric.

The output is a `DirectFeedbackPlan` consumed by `direct_refinement_orchestrator`,
which feeds the plan directly into the SSDBCODI provider. No pair-based
distance metric is learned on this path.

This is the single narrow boundary between dashboard-facing feedback and the
SSDBCODI algorithm's native semi-supervised inputs (seeds, `n_clusters`,
contamination, labeled outliers, and feature scaling).

## Why This Module Exists

SSDBCODI is semi-supervised and density-based. Every cluster expansion starts
from a labeled seed, and every outlier decision can be overridden by a manual
label. This means the same user feedback that Path A turns into pair
constraints can, on Path B, be expressed as native SSDBCODI inputs:

| User feedback | Path A (metric_learning_adapter) | Path B (this module) |
|---|---|---|
| `assign_cluster` label | intra-label must-link pairs | new/updated cluster seed |
| `mark_outlier` label | not used for pairs in Phase 1 | labeled outlier seed |
| `mark_not_outlier` label | not used for pairs in Phase 1 | remove outlier override |
| `feature_weight` intent | `feature_scale` dict, then ITML pre-scale | same `feature_scale`, used as-is |
| `group_similar` intent | sampled must-link pairs | co-assign both groups to same cluster seed |
| `group_dissimilar` intent | sampled cannot-link pairs | assign each group to distinct seeds |
| `merge_clusters` intent | cross-cluster must-link pairs | relabel seeds of absorbed cluster(s) |
| `anchor_point` intent | anchor-to-target must-link | promote anchor to cluster seed |
| `ignore_cluster` intent | exclude cluster from pair generation | exclude cluster seeds this run |
| `split_cluster` intent | **deferred** (ITML cannot change k) | `n_clusters += 1`, add interior seeds |
| `reclassify_outlier` intent | **deferred** (metric cannot move across contamination threshold) | toggle labeling outlier override |

`split_cluster` and `reclassify_outlier` are therefore **native** on Path B
while remaining deferred on Path A.

## Responsibilities

1. Accept a `StructuredInstruction` and a labeling annotation snapshot.
2. Validate instruction source, status, and referenced point IDs.
3. Reject incomplete or non-actionable instructions (these should have been
   filtered upstream by `intent_instruction`).
4. Compile inputs into a `DirectFeedbackPlan`.
5. Report conflicts when feedback is self-contradictory (e.g., same point
   assigned to two different clusters).
6. Provide a Flask page for inspecting the plan from sample inputs.

## Not Responsible For

1. Parsing natural language.
2. Rendering chat UI.
3. Managing selection state.
4. Running clustering directly.
5. Running outlier detection directly.
6. Owning manual label state.
7. Learning a Mahalanobis metric (that is `metric_learning_adapter`).
8. Running refinement end-to-end (that is `direct_refinement_orchestrator`).

## Target Files

```text
app/modules/direct_feedback_adapter/
  __init__.py
  schemas.py
  plan_builder.py
  fixtures.py
  routes.py
  templates/direct_feedback_adapter/index.html

tests/modules/direct_feedback_adapter/
  test_plan_builder.py
  test_routes.py
```

## Pipeline

```text
labeling annotations        ──┐
StructuredInstruction       ──┼─► plan_builder ─► DirectFeedbackPlan
selection groups (read-only)──┘
```

The plan is then consumed by `direct_refinement_orchestrator`, which feeds it
to the SSDBCODI provider.

## DirectFeedbackPlan Schema

```json
{
  "plan_id": "plan_003",
  "seed_updates": [
    {"point_id": "p17", "cluster_id": "cluster_2", "source": "chat_intent", "intent_id": "c4"},
    {"point_id": "p42", "is_outlier": true,         "source": "labeling",    "intent_id": null}
  ],
  "feature_scale": {
    "petal_length": 2.0,
    "sepal_width": 0.5
  },
  "param_overrides": {
    "n_clusters": 4,
    "alpha": 0.4,
    "beta": 0.3,
    "contamination": 0.13,
    "min_pts": 3
  },
  "excluded_clusters": ["cluster_5"],
  "merged_cluster_groups": [
    {"absorb_into": "cluster_1", "absorb_from": ["cluster_2"]}
  ],
  "conflicts": [],
  "diagnostics": {
    "sources": ["labeling", "chat_intent"],
    "intent_counts": {
      "feature_weight": 2,
      "group_similar": 1,
      "split_cluster": 1
    }
  }
}
```

`feature_scale` maps feature name to multiplier. `1.0` means unchanged. Values
above 1 emphasize the feature, below 1 de-emphasize it, and `0.0` drops it.
The orchestrator applies `S = diag(feature_scale)` as `X' = X · S` before
SSDBCODI runs.

## Intent-to-Plan Mapping (Phase 1, B-path)

| Intent | Plan effect |
|--------|-------------|
| `feature_weight` | Add/overwrite entry in `feature_scale`. |
| `group_similar` | Assign both groups' points as `seed_updates` with the same `cluster_id` (pick an existing cluster or allocate a pending one). |
| `group_dissimilar` | Assign each group's points as `seed_updates` under distinct `cluster_id` values so density expansion separates them. |
| `merge_clusters` | Add an entry to `merged_cluster_groups` and relabel the absorbed cluster's seeds to the target cluster. |
| `anchor_point` | Add one `seed_updates` entry promoting the anchor point to the target cluster. |
| `ignore_cluster` | Add the cluster ID to `excluded_clusters`; seeds of that cluster are skipped this run. |
| `split_cluster` | Increment `param_overrides.n_clusters` by one (or by the requested delta) and add interior seeds pointing to new cluster IDs. |
| `reclassify_outlier` | Add a `seed_updates` entry with `is_outlier: true` (or `false` for `mark_not_outlier`-style reclassify); the orchestrator routes this through labeling's outlier override. |

Labeling annotations are merged after intent compilation:

1. `assign_cluster` annotations become `seed_updates` with the annotated
   `cluster_id`, overriding any conflicting chat-derived seed for that point.
2. `mark_outlier` annotations become `seed_updates` with `is_outlier: true`.
3. `mark_not_outlier` annotations become `seed_updates` with `is_outlier: false`
   and clear any existing outlier seed for that point.

Manual labels take priority over chat-derived seeds on the same point so the
user's direct actions cannot be overridden by a chat intent.

## Conflict Detection

Before returning, the builder runs these checks:

1. Same `point_id` listed twice in `seed_updates` with different `cluster_id`
   values - reported in `conflicts`.
2. Same `point_id` assigned a cluster seed and also marked as an outlier -
   reported in `conflicts`.
3. `excluded_clusters` containing a cluster that is also referenced in
   `seed_updates` or `merged_cluster_groups` - reported in `conflicts`.
4. `merged_cluster_groups` where the same cluster appears in both
   `absorb_into` and `absorb_from` across different entries - reported.

When conflicts exist the adapter returns `ok: false` with the `conflicts` list
so the chatbox can ask the user to resolve them.

## Flask Routes

```text
/modules/direct-feedback-adapter/                       adapter debug page
/modules/direct-feedback-adapter/health                 module health
/modules/direct-feedback-adapter/api/plan               build DirectFeedbackPlan from inputs
/modules/direct-feedback-adapter/api/preview            preview what SSDBCODI inputs the plan would produce
```

## Flask Debug Page Requirements

The page should show:

1. editable JSON textareas for sample `StructuredInstruction` and labeling
   annotations.
2. generated `DirectFeedbackPlan` with seed updates, feature_scale, param
   overrides, and conflict list.
3. a table view of which input intents produced which plan entries.
4. explicit note that `split_cluster` and `reclassify_outlier` are supported
   on this path (unlike Path A).
5. link to `/workflows/instruction-ssdbcodi/` for end-to-end testing.

## Testing

Unit tests (plan_builder):

1. `feature_weight` populates `feature_scale`, not seeds.
2. `group_similar` produces `seed_updates` with a shared `cluster_id`.
3. `group_dissimilar` produces `seed_updates` with distinct `cluster_id`s.
4. `merge_clusters` produces a `merged_cluster_groups` entry and relabels
   seeds.
5. `anchor_point` produces a single seed update for the anchor.
6. `ignore_cluster` adds the cluster to `excluded_clusters`.
7. `split_cluster` increments `n_clusters` in `param_overrides` and adds
   interior seeds.
8. `reclassify_outlier` produces a seed update with the correct `is_outlier`
   flag.
9. Labeling `assign_cluster` annotations override conflicting chat-derived
   seeds on the same point.
10. Same point with contradictory seed assignments is reported in
    `conflicts`.
11. Same point assigned to a cluster and marked as outlier is reported in
    `conflicts`.

Flask route tests:

1. debug page returns 200.
2. plan API returns `DirectFeedbackPlan` for valid input.
3. plan API returns conflict list for self-contradictory input.
4. plan API accepts `split_cluster` and `reclassify_outlier` inputs without
   `intent_deferred` errors.

Manual browser check:

1. open `/modules/direct-feedback-adapter/`.
2. submit a sample instruction with `split_cluster` and confirm the plan
   contains an `n_clusters += 1` override plus new seeds.
3. submit a sample instruction with `reclassify_outlier` and confirm the plan
   contains the appropriate outlier seed update.
4. submit a sample instruction with contradictory seed assignments and
   confirm the conflict list is populated.

## Completion Criteria

This module is complete when:

1. Plan building is testable on both chat-derived and labeling-derived input.
2. All Phase 1 intents plus `split_cluster` and `reclassify_outlier` are
   supported end to end.
3. Conflict detection is tested with at least three distinct conflict shapes.
4. The debug page makes the plan inspectable in the browser.
5. `/workflows/instruction-ssdbcodi/` can exercise the adapter with real
   selection and labeling state.
